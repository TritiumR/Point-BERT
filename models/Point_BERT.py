import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
from .dvae import Group
from .dvae import DiscreteVAE, Encoder

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            # print('channel: ', last_channel, out_channel)
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
            # print('new_points: ', new_points.shape)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 384, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 384),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q


class Critic(nn.Module):
    def __init__(self, feat_dim):
        super(Critic, self).__init__()

        self.propagation_2 = PointNetFeaturePropagation(in_channel=feat_dim + 3, mlp=[feat_dim * 4, feat_dim])
        self.propagation_1 = PointNetFeaturePropagation(in_channel=feat_dim + 3, mlp=[feat_dim * 4, feat_dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel=feat_dim + 3, mlp=[feat_dim * 4, feat_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k=4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k=4)

        self.mlp1 = nn.Linear(feat_dim + 3 + 3, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, 1)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, pts, center, feature_list, dirs1, dirs2):
        center_level_0 = pts.transpose(-1, -2).contiguous()
        f_level_0 = center_level_0
        center_level_1 = fps(pts, 2048).transpose(-1, -2).contiguous()
        f_level_1 = center_level_1
        center_level_2 = fps(pts, 1024).transpose(-1, -2).contiguous()
        f_level_2 = center_level_2
        center_level_3 = center.transpose(-1, -2).contiguous()

        # up sampling
        f_level_3 = feature_list
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list)
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list)

        # bottom up
        f_level_2 = self.dgcnn_pro_2.forward(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1.forward(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 = self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

        # print(f_level_0.shape)

        # pixel feature
        pixel_feats = f_level_0[:, :, 0]

        input_queries = torch.cat([dirs1, dirs2], dim=1)
        net = torch.cat([pixel_feats, input_queries], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net).squeeze(-1)
        return net

    # cross entropy loss
    def get_ce_loss(self, pred_logits, gt_labels):
        loss = self.BCELoss(pred_logits, gt_labels.float())
        return loss

    def inference_whole_pc(self, pts, center, feature_list, dirs1, dirs2):
        center_level_0 = pts.transpose(-1, -2).contiguous()
        f_level_0 = center_level_0
        center_level_1 = fps(pts, 2048).transpose(-1, -2).contiguous()
        f_level_1 = center_level_1
        center_level_2 = fps(pts, 1024).transpose(-1, -2).contiguous()
        f_level_2 = center_level_2
        center_level_3 = center.transpose(-1, -2).contiguous()

        # up sampling
        f_level_3 = feature_list
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list)
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list)

        # bottom up
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 = self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

        # print(f_level_0.shape)
        feats = f_level_0

        num_pts = feats.shape[-1]
        batch_size = feats.shape[0]

        feats = feats.permute(0, 2, 1)  # B x N x F
        feats = feats.reshape(batch_size * num_pts, -1)

        input_queries = torch.cat([dirs1, dirs2], dim=-1)
        input_queries = input_queries.unsqueeze(dim=1).repeat(1, num_pts, 1)
        input_queries = input_queries.reshape(batch_size * num_pts, -1)

        net = torch.cat([feats, input_queries], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net).squeeze(-1)
        pred_result_logits = net

        soft_pred_results = torch.sigmoid(pred_result_logits)
        soft_pred_results = soft_pred_results.reshape(batch_size, num_pts)

        return soft_pred_results


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100


    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]


        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')


    def forward(self, pts):
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)
        ret = self.cls_head_finetune(concat_f)
        return ret

    def forward_encode(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        feature_list = self.blocks(x, pos)
        # print("feature_list: ", feature_list.shape)
        feature_list = self.norm(feature_list)[:, 1:, :].transpose(-1, -2).contiguous()

        return feature_list, center


class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate 
        self.cls_dim = config.transformer_config.cls_dim 
        self.replace_pob = config.transformer_config.replace_pob
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[Transformer args] {config.transformer_config}', logger = 'dVAE BERT')
        # define the encoder
        self.encoder_dims =  config.dvae_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        try:
            self.mask_rand = config.mask_rand
        except:
            self.mask_rand = False
        
        # define the learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # pos embedding for each patch 
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)
        # head for token classification
        self.num_tokens = config.dvae_config.num_tokens
        self.lm_head = nn.Linear(self.trans_dim, self.num_tokens)
        # head for cls contrast
        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )  
        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _prepare_encoder(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        encoder_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items() if 'encoder' in k}

        self.encoder.load_state_dict(encoder_ckpt, strict=True)
        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger = 'dVAE BERT')


    def _mask_center(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0) # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p =2 ,dim = -1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0] # G
            ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos


    def _random_replace(self, group_input_tokens, bool_masked_pos, noaug = False):
        '''
            group_input_tokens : B G C
            bool_masked_pos : B G  
            -----------------
            replaced_group_input_tokens: B G C
        '''
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_tokens, bool_masked_pos
        
        replace_mask = (torch.rand(group_input_tokens.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  #  True for flake input

        detached_group_input_tokens = group_input_tokens.detach()
        flatten_group_input_tokens = detached_group_input_tokens.reshape(detached_group_input_tokens.size(0) * detached_group_input_tokens.size(1), detached_group_input_tokens.size(2))
        idx = torch.randperm(flatten_group_input_tokens.shape[0])
        shuffled_group_input_tokens = flatten_group_input_tokens[idx].reshape(detached_group_input_tokens.size(0), detached_group_input_tokens.size(1), detached_group_input_tokens.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_tokens)
        replaced_group_input_tokens = group_input_tokens * (1 - replace_mask) + shuffled_group_input_tokens * replace_mask
        return replaced_group_input_tokens, overall_mask

    def forward(self, neighborhood, center, return_all_tokens = False, only_cls_tokens = False, noaug = False):
        # generate mask
        if self.mask_rand:
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center(center, noaug = noaug) # B G
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # replace the point
        replaced_group_input_tokens, overall_mask = self._random_replace(group_input_tokens, bool_masked_pos.clone(), noaug = noaug)
        batch_size, seq_len, _ = replaced_group_input_tokens.size()
        # prepare cls and mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)  
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # mask the input tokens
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        maksed_group_input_tokens = replaced_group_input_tokens * (1 - w) + mask_token * w
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, maksed_group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        # only return the cls feature, for moco contrast
        if only_cls_tokens:
            return self.cls_head(x[:, 0])
        logits = self.lm_head(x[:, 1:])
        if return_all_tokens:
            return self.cls_head(x[:, 0]), logits
        else:
            # return the flake logits
            return self.cls_head(x[:, 0]), logits[~overall_mask], logits[overall_mask], overall_mask # reduce the Batch dim        

@MODELS.register_module()
class Point_BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_BERT] build dVAE_BERT ...', logger ='Point_BERT')
        self.config = config
        self.m = config.m
        self.T = config.T
        self.K = config.K

        self.moco_loss = config.transformer_config.moco_loss
        self.dvae_loss = config.transformer_config.dvae_loss
        self.cutmix_loss = config.transformer_config.cutmix_loss

        self.return_all_tokens = config.transformer_config.return_all_tokens
        if self.return_all_tokens:
            print_log(f'[Point_BERT] Point_BERT calc the loss for all token ...', logger ='Point_BERT')
        else:
            print_log(f'[Point_BERT] Point_BERT [NOT] calc the loss for all token ...', logger ='Point_BERT')
        
        self.transformer_q = MaskTransformer(config)
        self.transformer_q._prepare_encoder(self.config.dvae_config.ckpt)
        
        self.transformer_k = MaskTransformer(config)
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.dvae = DiscreteVAE(config.dvae_config)
        self._prepare_dvae()

        for param in self.dvae.parameters():
            param.requires_grad = False

        self.group_size = config.dvae_config.group_size
        self.num_group = config.dvae_config.num_group

        print_log(f'[Point_BERT Group] cutmix_BERT divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_BERT')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # create the queue
        self.register_buffer("queue", torch.randn(self.transformer_q.cls_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # loss
        self.build_loss_func()

    def _prepare_dvae(self):
        dvae_ckpt = self.config.dvae_config.ckpt
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger ='Point_BERT')


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none')

    def forward_eval(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            cls_feature = self.transformer_q(neighborhood, center, only_cls_tokens = True, noaug = True)
            return cls_feature

    def _mixup_pc(self, neighborhood, center, dvae_label):
        '''
            neighborhood : B G M 3
            center: B G 3
            dvae_label: B G
            ----------------------
            mixup_ratio: /alpha:
                mixup_label = alpha * origin + (1 - alpha) * flip

        '''
        mixup_ratio = torch.rand(neighborhood.size(0))
        mixup_mask = torch.rand(neighborhood.shape[:2]) < mixup_ratio.unsqueeze(-1)
        mixup_mask = mixup_mask.type_as(neighborhood)
        mixup_neighborhood = neighborhood * mixup_mask.unsqueeze(-1).unsqueeze(-1) + neighborhood.flip(0) * (1 - mixup_mask.unsqueeze(-1).unsqueeze(-1))
        mixup_center = center * mixup_mask.unsqueeze(-1) + center.flip(0) * (1 - mixup_mask.unsqueeze(-1))
        mixup_dvae_label = dvae_label * mixup_mask + dvae_label.flip(0) * (1 - mixup_mask)

        return mixup_ratio.to(neighborhood.device), mixup_neighborhood, mixup_center, mixup_dvae_label.long()


    def forward(self, pts, noaug = False, **kwargs):
        if noaug:
            return self.forward_eval(pts)
        else:
            # divide the point cloud in the same form. This is important
            neighborhood, center = self.group_divider(pts)
            # produce the gt point tokens
            with torch.no_grad():
                gt_logits = self.dvae.encoder(neighborhood) 
                gt_logits = self.dvae.dgcnn_1(gt_logits, center) #  B G N
                dvae_label = gt_logits.argmax(-1).long() # B G 
            # forward the query model in mask style 1.
            if self.return_all_tokens:
                q_cls_feature, logits = self.transformer_q(neighborhood, center, return_all_tokens = self.return_all_tokens) # logits :  N G C 
            else:
                q_cls_feature, real_logits, flake_logits, mask = self.transformer_q(neighborhood, center, return_all_tokens = self.return_all_tokens) # logits :  N' C where N' is the mask.sum() 
            q_cls_feature = nn.functional.normalize(q_cls_feature, dim=1)

            mixup_ratio, mixup_neighborhood, mixup_center, mix_dvae_label = self._mixup_pc(neighborhood, center, dvae_label)
            if self.return_all_tokens:
                mixup_cls_feature, mixup_logits = self.transformer_q(mixup_neighborhood, mixup_center, return_all_tokens = self.return_all_tokens) 
            else:
                mixup_cls_feature, mixup_real_logits, mixup_flake_logits, mixup_mask = self.transformer_q(mixup_neighborhood, mixup_center, return_all_tokens = self.return_all_tokens)
            mixup_cls_feature = nn.functional.normalize(mixup_cls_feature, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                k_cls_feature = self.transformer_k(neighborhood, center, only_cls_tokens = True)  # keys: NxC
                k_cls_feature = nn.functional.normalize(k_cls_feature, dim=1)

            if self.moco_loss:
                # ce loss with moco contrast
                l_pos = torch.einsum('nc, nc->n', [q_cls_feature, k_cls_feature]).unsqueeze(-1) # n 1 
                l_neg = torch.einsum('nc, ck->nk', [q_cls_feature, self.queue.clone().detach()]) # n k
                ce_logits = torch.cat([l_pos, l_neg], dim=1)
                ce_logits /= self.T
                labels = torch.zeros(l_pos.shape[0], dtype=torch.long).to(pts.device)
                moco_loss = self.loss_ce(ce_logits, labels)
            else:
                moco_loss = torch.tensor(0.).to(pts.device)

            if self.dvae_loss:
                if self.return_all_tokens:
                    dvae_loss = self.loss_ce(logits.reshape(-1, logits.size(-1)), dvae_label.reshape(-1,)) + \
                             self.loss_ce(mixup_logits.reshape(-1, mixup_logits.size(-1)), mix_dvae_label.reshape(-1,))
                else:
                    dvae_loss = self.loss_ce(flake_logits, dvae_label[mask]) + \
                        self.loss_ce(mixup_flake_logits, mix_dvae_label[mixup_mask])
            else:
                dvae_loss = torch.tensor(0.).to(pts.device)

            if self.cutmix_loss:                
                l_pos = torch.einsum('nc, mc->nm', [mixup_cls_feature, k_cls_feature]) # n n
                l_neg = torch.einsum('nc, ck->nk', [mixup_cls_feature, self.queue.clone().detach()]) # n k
                ce_logits = torch.cat([l_pos, l_neg], dim=1)
                ce_logits /= self.T
                labels = torch.arange(l_pos.shape[0], dtype=torch.long).to(pts.device)
                cutmix_loss = (mixup_ratio * self.loss_ce_batch(ce_logits, labels) + (1 - mixup_ratio) * self.loss_ce_batch(ce_logits, labels.flip(0))).mean()
            else:
                cutmix_loss = torch.tensor(0.).to(pts.device)
            self._dequeue_and_enqueue(k_cls_feature)
            return moco_loss + dvae_loss, cutmix_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
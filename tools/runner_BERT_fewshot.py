import os
import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from models.Point_BERT import Critic
from datagen import DataGen
from data import SAPIENVisionDataset
from aff_utils import worker_init_fn, collate_feats, force_mkdir
import shutil
import torch.nn.functional as F
import torch.utils.data

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        
def run_net(args, config, train_writer=None, val_writer=None):
    # create training and validation datasets and data loaders
    data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction',
                     'gripper_forward_direction',
                     'result', 'cur_dir', 'shape_id', 'cnt_id', 'trial_id', 'is_original', 'obj_state', 'category',
                     'camera_direction']

    logger = get_logger(args.log_name)

    # create models
    decoder = Critic(384).to(args.local_rank)

    # create optimizers
    network_opt = torch.optim.Adam(decoder.parameters(), lr=0.001, weight_decay=1e-5)

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=5000, gamma=0.9)

    args.exp_name = f'exp-BERT-{args.primact_type}-{args.category_types}-{args.exp_name}'

    args.data_dir = args.data_dir_prefix + '/' + args.exp_name
    args.exp_dir = os.path.join('./logs', args.exp_name)

    if os.path.exists(args.exp_dir):
        shutil.rmtree(args.exp_dir)
    os.mkdir(args.exp_dir)
    os.mkdir(os.path.join(args.exp_dir, 'ckpts'))

    if args.category_types is None:
        args.category_types = ['Box', 'Door', 'Faucet', 'Kettle', 'Microwave', 'Refrigerator', 'StorageFurniture',
                               'Switch', 'TrashCan', 'Window']
    else:
        args.category_types = args.category_types.split(',')
    print('category_types: %s' % str(args.category_types))

    with open(os.path.join(args.offline_data_dir, 'data_tuple_list.txt'), 'r') as fin:
        all_train_data_list = [os.path.join(args.offline_data_dir, l.rstrip()) for l in fin.readlines()]

    ruler_data_list = []
    for item in all_train_data_list:
        if int(item.split('_')[-1]) < args.num_interaction_data_offline:
            ruler_data_list.append(item)
    print('len(train_data_list): %d' % len(ruler_data_list))

    # prepare data_dir
    if os.path.exists(args.data_dir):
        shutil.rmtree(args.data_dir)
    os.mkdir(args.data_dir)

    # load dataset
    ruler_dataset = SAPIENVisionDataset([args.primact_type], args.category_types, data_features, args.buffer_max_num,
                                        abs_thres=0.01, rel_thres=0.5, dp_thres=0.5,
                                        img_size=224, no_true_false_equal=True)

    ruler_dataset.load_data(ruler_data_list)

    ruler_dataloader = torch.utils.data.DataLoader(ruler_dataset, batch_size=args.batch_size, shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=0, drop_last=True, collate_fn=collate_feats,
                                                   worker_init_fn=worker_init_fn)

    train_dataset = SAPIENVisionDataset([args.primact_type], args.category_types, data_features, args.buffer_max_num,
                                        abs_thres=0.01, rel_thres=0.5, dp_thres=0.5,
                                        img_size=224, no_true_false_equal=args.no_true_false_equal)

    # create a data generator
    datagen = DataGen(10, None)

    # create a trial counter
    trial_counter = dict()

    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0

    if args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts)
    else:
        print_log('Training from scratch', logger=logger)

    base_model.to(args.local_rank)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, args.epoch + 1):
        base_model.eval()

        sample_conf_list = []
        sample_conf_name = []
        sample_conf_value = []

        cur_sample_conf_dir = os.path.join(args.data_dir, 'epoch-%04d_sample' % epoch)
        force_mkdir(cur_sample_conf_dir)
        ruler_batches = enumerate(ruler_dataloader, 0)
        for ruler_batch_ind, batch in ruler_batches:
            decoder.eval()
            with torch.no_grad():
                categories = batch[data_features.index('category')]
                input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(args.local_rank)  # B x 3N x 3
                input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(args.local_rank)  # B x 3N x 2
                input_movables = torch.cat(batch[data_features.index('pc_movables')], dim=0).to(args.local_rank)  # B x 3N
                batch_size = input_pcs.shape[0]

                input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, args.num_point_per_shape).long().reshape(-1)  # BN
                fps_idx = pointnet2_utils.furthest_point_sample(input_pcs, args.num_point_per_shape)  # BN
                input_pcid2 = fps_idx.long().reshape(-1)  # BN
                input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)
                whole_pxids = input_pxids[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)
                whole_movables = input_movables[input_pcid1, input_pcid2].reshape(batch_size, args.num_point_per_shape)

                # sample a random EE orientation
                random_up = torch.randn(args.batch_size, 3).float().to(args.local_rank)
                random_forward = torch.randn(args.batch_size, 3).float().to(args.local_rank)
                random_left = torch.cross(random_up, random_forward)
                random_forward = torch.cross(random_left, random_up)
                random_dirs1 = F.normalize(random_up, dim=1).float()
                random_dirs2 = F.normalize(random_forward, dim=1).float()

                # forward through the transformer to encode feature
                # fps_idx = fps_idx[:, np.random.choice(args.num_point_per_shape, args.num_point_per_shape, False)]
                # points = pointnet2_utils.gather_operation(input_pcs.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                # print('points_shape1: ', points.shape)
                # points = train_transforms(points)
                # print('points_shape2: ', points.shape)
                # print(input_pcs.shape)
                whole_feats, center = base_model.forward_encode(input_pcs)  # B x 2, B x F x N
                # print('whole[0]_shape: ', whole_feats[1].shape)

                # test over the entire image
                whole_pc_scores1 = decoder.inference_whole_pc(input_pcs, center, whole_feats, random_dirs1, random_dirs2)  # B x N
                whole_pc_scores2 = decoder.inference_whole_pc(input_pcs, center, whole_feats, -random_dirs1, random_dirs2)  # B x N

                # add to the sample_conf_list if wanted
                ss_cur_dir = batch[data_features.index('cur_dir')]
                ss_shape_id = batch[data_features.index('shape_id')]
                ss_obj_state = batch[data_features.index('obj_state')]
                ss_is_original = batch[data_features.index('is_original')]

                for i in range(args.batch_size):
                    key = ss_shape_id[i] + '-' + ss_obj_state[i]
                    # print("22222")
                    if (ss_is_original[i]) and (key not in sample_conf_name):
                        # print("333333")
                        sample_conf_name.append(key)
                        if ss_shape_id[i] not in trial_counter.keys():
                            trial_counter[ss_shape_id[i]] = args.num_interaction_data_offline

                        gt_movable = whole_movables[i].cpu().numpy()

                        whole_pc_score1 = whole_pc_scores1[i].cpu().numpy() * gt_movable
                        whole_pc_confidence1 = abs(2 * whole_pc_score1 - 1)
                        whole_pc_confidence_sum1 = np.sum(whole_pc_confidence1)
                        whole_pc_confidence_min1 = np.min(whole_pc_confidence1)
                        whole_pc_confidence_max1 = np.max(whole_pc_confidence1)

                        whole_pc_score2 = whole_pc_scores2[i].cpu().numpy() * gt_movable
                        whole_pc_confidence2 = abs(2 * whole_pc_score2 - 1)
                        whole_pc_confidence_sum2 = np.sum(whole_pc_confidence2)
                        whole_pc_confidence_min2 = np.min(whole_pc_confidence2)
                        whole_pc_confidence_max2 = np.max(whole_pc_confidence2)

                        random_dir1 = random_dirs1[i].cpu().numpy()
                        random_dir2 = random_dirs2[i].cpu().numpy()

                        # sample <X, Y> on each img
                        if whole_pc_confidence_sum1 < whole_pc_confidence_sum2:
                            ptid = np.argmin(whole_pc_confidence1)
                        else:
                            ptid = np.argmin(whole_pc_confidence2)

                        whole_pc_confidence_min_sum = min(whole_pc_confidence_sum1, whole_pc_confidence_sum2)

                        X = whole_pxids[i, ptid, 0].item()
                        Y = whole_pxids[i, ptid, 1].item()

                        # add job to the queue
                        str_cur_dir1 = ',' + ','.join(['%f' % elem for elem in random_dir1])
                        str_cur_dir2 = ',' + ','.join(['%f' % elem for elem in random_dir2])

                        sample_conf_list.append((args.offline_data_dir, str_cur_dir1, str_cur_dir2,
                                                 ss_cur_dir[i].split('/')[-1], cur_sample_conf_dir, X, Y))

                        sample_conf_value.append(whole_pc_confidence_min_sum / args.num_point_per_shape)

                        print(f'!!!!!!!! check category {categories[i]} shape {ss_shape_id[i]} state {ss_obj_state[i]} !!!!!!')

                        # print('sum1: ', whole_pc_confidence_sum1)
                        # print('sum2: ', whole_pc_confidence_sum2)
                        # print('min1: ', whole_pc_confidence_min1)
                        # print('min2: ', whole_pc_confidence_min2)
                        print('min_sum: ', whole_pc_confidence_min_sum / args.num_point_per_shape)
                        print('min-max ', min(whole_pc_confidence_min1, whole_pc_confidence_min2),
                              max(whole_pc_confidence_max1, whole_pc_confidence_max2))

        sample_conf_value = np.array(sample_conf_value)
        sample_idx = np.argsort(sample_conf_value)[: args.sample_max_num]
        for idx in sample_idx:
            item = sample_conf_list[idx]
            shape_id, category, _, _, _ = item[3].split('_')
            datagen.add_one_recollect_job(item[0], item[1], item[2], item[3], item[4], item[5], item[6],
                                          args.num_interaction_data_offline * (epoch + 1))
            state = sample_conf_name[idx].split('-')[-1]
            trial_counter[shape_id] += 1
            print('category %s shape %s state %s confidence %d', category, shape_id, state, sample_conf_value[idx])

        torch.save(sample_conf_list, os.path.join(args.exp_dir, 'ckpts', '%d-sample_conf_list.pth' % epoch))

        # start all jobs
        datagen.start_all()
        print(f'  [ Started generating epoch-{epoch} data ]')
        train_data_list = datagen.join_all()
        print(f'  [ Gathered epoch-{epoch} data ]')

        cur_data_folders = []
        for item in train_data_list:
            item = '/'.join(item.split('/')[:-1])
            if item not in cur_data_folders:
                cur_data_folders.append(item)
        for cur_data_folder in cur_data_folders:
            with open(os.path.join(cur_data_folder, 'data_tuple_list.txt'), 'w') as fout:
                for item in train_data_list:
                    if cur_data_folder == '/'.join(item.split('/')[:-1]):
                        fout.write(item.split('/')[-1] + '\n')

        train_dataset.load_data(train_data_list)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       pin_memory=True,
                                                       num_workers=0, drop_last=True, collate_fn=collate_feats,
                                                       worker_init_fn=worker_init_fn)
        train_num_batch = len(train_dataloader)
        print('train_num_batch: ', train_num_batch)

        base_model.eval()  # set model to testing mode

        for batch_ind, batch in enumerate(train_dataloader):
            # save checkpoint
            if batch_ind == 0:
                with torch.no_grad():
                    print('Saving checkpoint ...... ')
                    torch.save(decoder.state_dict(), os.path.join(args.exp_dir, 'ckpts', '%d-decoder.pth' % epoch))
                    torch.save(network_opt.state_dict(), os.path.join(args.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
                    torch.save(network_lr_scheduler.state_dict(),
                               os.path.join(args.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
                    torch.save(train_dataset, os.path.join(args.exp_dir, 'ckpts', '%d-train_dataset.pth' % epoch))
                    print('DONE')

            with torch.no_grad():
                # shape_id = batch[data_features.index('shape_id')]
                # cnt_id = batch[data_features.index('cnt_id')]
                # trial_id = batch[data_features.index('trial_id')]
                # categories = batch[data_features.index('category')]
                # print(shape_id)
                # print(trial_id)
                # prepare input
                input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(args.local_rank)  # B x 3N x 3
                # input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(args.local_rank)  # B x 3N x 2
                # input_movables = torch.cat(batch[data_features.index('pc_movables')], dim=0).to(args.local_rank)  # B x 3N
                batch_size = input_pcs.shape[0]

                input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, args.num_point_per_shape).long().reshape(-1)  # BN
                fps_idx = pointnet2_utils.furthest_point_sample(input_pcs, args.num_point_per_shape)  # BN
                input_pcid2 = fps_idx.long().reshape(-1)  # BN
                input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)
                # input_pxids = input_pxids[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)
                # input_movables = input_movables[input_pcid1, input_pcid2].reshape(batch_size, args.num_point_per_shape)

                input_dirs1 = torch.cat(batch[data_features.index('gripper_direction')], dim=0).to(args.local_rank)  # B x 3
                input_dirs2 = torch.cat(batch[data_features.index('gripper_forward_direction')], dim=0).to(args.local_rank)  # B x 3

                # forward through the transformer to encode feature
                # fps_idx = fps_idx[:, np.random.choice(args.num_point_per_shape, args.num_point_per_shape, False)]
                # points = pointnet2_utils.gather_operation(input_pcs.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                # print('points_shape1: ', points.shape)
                # points = train_transforms(points)
                # print('points_shape2: ', points.shape)
                whole_feats, center = base_model.forward_encode(input_pcs)  # B x 2, B x F x N

            decoder.train()

            pred_result_logits = decoder.forward(input_pcs, center, whole_feats, input_dirs1, input_dirs2)

            gt_result = torch.Tensor(batch[data_features.index('result')]).long().to(args.local_rank)  # B

            result_loss_per_data = decoder.get_ce_loss(pred_result_logits, gt_result)

            loss = result_loss_per_data.mean()

            print('epoch', epoch, 'batch_id', batch_ind, 'result', loss.item())

            # optimize one step for critic
            network_opt.zero_grad()
            loss.backward()
            network_opt.step()
            network_lr_scheduler.step()

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def test_net(args, config):
    data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction',
                     'gripper_forward_direction',
                     'result', 'cur_dir', 'shape_id', 'cnt_id', 'trial_id', 'is_original', 'obj_state', 'category',
                     'camera_direction']

    args.exp_name = f'exp-BERT-{args.primact_type}-{args.category_types}-{args.exp_name}'

    args.exp_dir = os.path.join('./logs', args.exp_name)

    if args.category_types is None:
        args.category_types = ['Box', 'Door', 'Faucet', 'Kettle', 'Microwave', 'Refrigerator', 'StorageFurniture',
                               'Switch', 'TrashCan', 'Window']
    else:
        args.category_types = args.category_types.split(',')
    print('category_types: %s' % str(args.category_types))

    with open(os.path.join(args.offline_data_dir, 'data_tuple_list.txt'), 'r') as fin:
        all_train_data_list = [os.path.join(args.offline_data_dir, l.rstrip()) for l in fin.readlines()]

    test_data_list = []
    for item in all_train_data_list:
        if int(item.split('_')[-1]) < args.num_interaction_data_offline:
            test_data_list.append(item)
    print('len(train_data_list): %d' % len(test_data_list))

    # load dataset
    test_dataset = SAPIENVisionDataset([args.primact_type], args.category_types, data_features, args.buffer_max_num,
                                        abs_thres=0.01, rel_thres=0.5, dp_thres=0.5,
                                        img_size=224, no_true_false_equal=True)

    test_dataset.load_data(test_data_list)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=0, drop_last=True, collate_fn=collate_feats,
                                                   worker_init_fn=worker_init_fn)

    # build model
    base_model = builder.model_builder(config.model)

    if args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts)
    else:
        print_log('Training from scratch', logger=logger)

    base_model.to(args.local_rank)

    # create models
    decoder = Critic(384)

    base_model.to(args.local_rank)
    decoder.to(args.local_rank)

    for model_epoch in range(args.start_epoch, args.epoch + 1):
        base_model.eval()
        data_to_restore = torch.load(os.path.join(args.exp_dir, 'ckpts', '%d-decoder.pth' % model_epoch))
        decoder.load_state_dict(data_to_restore)
        print('loading model from: ', os.path.join(args.exp_dir, 'ckpts', '%d-decoder.pth' % model_epoch))
        decoder.eval()

        total_pred_and_gt = 0
        total_pred_sum = 1
        total_gt_sum = 1

        for batch_ind, batch in enumerate(test_dataloader):
            with torch.no_grad():
                input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(args.local_rank)  # B x 3N x 3
                batch_size = input_pcs.shape[0]

                input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, args.num_point_per_shape).long().reshape(-1)  # BN
                fps_idx = pointnet2_utils.furthest_point_sample(input_pcs, args.num_point_per_shape)  # BN
                input_pcid2 = fps_idx.long().reshape(-1)  # BN
                input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)

                input_dirs1 = torch.cat(batch[data_features.index('gripper_direction')], dim=0).to(args.local_rank)  # B x 3
                input_dirs2 = torch.cat(batch[data_features.index('gripper_forward_direction')], dim=0).to(args.local_rank)  # B x 3

                whole_feats, center = base_model.forward_encode(input_pcs)  # B x 2, B x F x N

                pred_result_logits = decoder.forward(input_pcs, center, whole_feats, input_dirs1, input_dirs2)

                # prepare gt
                gt_result = torch.Tensor(batch[data_features.index('result')]).long().numpy()  # B

                # compute correctness
                pred_results = pred_result_logits.detach().cpu().numpy() > 0
                pred_and_gt = np.logical_and(pred_results, gt_result)

                total_pred_and_gt += pred_and_gt.sum()
                total_pred_sum += pred_results.sum()
                total_gt_sum += gt_result.sum()

        precise = total_pred_and_gt / total_pred_sum
        recall = total_pred_and_gt / total_gt_sum
        F1 = 2 * (precise * recall) / (precise + recall)

        precise = round(precise, 3)
        recall = round(recall, 3)
        F1 = round(F1, 3)
        print('score from: ', args.exp_dir + f'-{model_epoch}')
        print('Score:  ', precise, '   ', recall, '   ', F1)

        with open(os.path.join(args.exp_dir, 'F1_score.txt'), 'a') as fout:
            if model_epoch == args.start_epoch:
                fout.write(f'{args.offline_data_dir} Score: \n')
            fout.write(f'{model_epoch}   {precise}   {recall}   {F1}\n')


def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger=logger)
        acc = 0.
        for time in range(1, 10):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc

        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)


def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    print_log('[TEST] acc = %.4f' % acc, logger=logger)

    return acc

categories="Bucket Table Door TrashCan Refrigerator WashingMachine Microwave"
for category in $categories
do
  bash ./scripts/train_BERT.sh 3 \
    --config cfgs/ModelNet_models/PointTransformer_8192point.yaml \
    --finetune_model \
    --ckpts ./Point-BERT.pth \
    --exp_name few-shot_"$category" \
    --num_interaction_data_offline 6 \
    --batch_size 10 \
    --num_point_per_shape 8192 \
    --offline_data_dir ./confidence_data/gt_data-train_fixed_cam_new_new_ruler_data-pulling \
    --category_types "$category" \
    --primact_type pulling \
    --buffer_max_num 10000 \
    --epoch 30 \
    --sample_max_num 10 \
    --data_dir_prefix ./confidence_data
done


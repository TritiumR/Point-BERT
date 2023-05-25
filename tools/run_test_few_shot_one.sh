categories="Faucet Window"
for category in $categories
do
  bash ./scripts/train_BERT.sh 1 \
    --config cfgs/ModelNet_models/PointTransformer_8192point.yaml \
    --test \
    --ckpts ./Point-BERT.pth \
    --exp_name few-shot_"$category" \
    --num_interaction_data_offline 100 \
    --batch_size 16 \
    --num_point_per_shape 8192 \
    --offline_data_dir ./confidence_data/gt_data-train_fixed_cam_new_new_test_data-pushing \
    --category_types "$category" \
    --primact_type pushing \
    --buffer_max_num 200000 \
    --start_epoch 0 \
    --no_true_false_equal \
    --epoch 20 \
    --data_dir_prefix ./confidence_data
done


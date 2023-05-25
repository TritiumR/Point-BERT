categories="Faucet Window"
for category in $categories
do
  bash ./scripts/train_BERT.sh 1 \
    --config cfgs/ModelNet_models/PointTransformer_8192point.yaml \
    --finetune_model \
    --ckpts ./Point-BERT.pth \
    --exp_name few-shot_"$category"_pulling \
    --num_interaction_data_offline 6 \
    --batch_size 16 \
    --num_point_per_shape 8192 \
    --offline_data_dir ./confidence_data/gt_data-train_fixed_cam_new_new_ruler_data-pulling \
    --additional_data_dir ./confidence_data/gt_data-train_fixed_cam_train_data-pulling \
    --additional_category_types StorageFurniture \
    --additional_num_interaction 1 \
    --category_types "$category" \
    --primact_type pulling \
    --buffer_max_num 100000 \
    --epoch 20 \
    --sample_max_num 10 \
    --data_dir_prefix ./confidence_data
done


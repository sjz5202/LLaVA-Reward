# phi3 BT lora
deepspeed train_llava_reward.py \
     --save_path /code/ckpt/test/phi3_test_gpm_lora \
     --save_steps 2 \
     --logging_steps 1 \
     --eval_steps 10000 \
     --accumulated_gradient 1 \
     --micro_train_batch_size 1 \
     --pretrain microsoft/Phi-3.5-vision-instruct \
     --bf16 \
     --max_epochs 3 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 2e-4 \
     --general_preference_tau 0.1 \
     --dataset /code/LLaVA-Reward/data/ImageReward/alignment/imagereward_alignment_no_lap_85.json \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --group_size 1 \
     --value_head_dim 1 \
     --save_best_model 2 \
     --train_split_ratio 1 \
     --freeze_vision_model \
     --lora_rank 128 \
     --lora_alpha 256 \
     --ft_projector \
     --add_cross_attention \
     --lora_dropout 0.05

# # phi3 gpm lora
# deepspeed train_llava_reward.py \
#      --save_path /scr/shijie/hf/rewardmodel/test/phi3_test_gpm_lora \
#      --save_steps 2 \
#      --logging_steps 1 \
#      --eval_steps 10000 \
#      --accumulated_gradient 4 \
#      --micro_train_batch_size 4 \
#      --pretrain microsoft/Phi-3.5-vision-instruct \
#      --bf16 \
#      --max_epochs 3 \
#      --max_len 2048 \
#      --zero_stage 3 \
#      --learning_rate 2e-4 \
#      --general_preference_tau 0.1 \
#      --dataset data/open_preference/test_rest.json \
#      --dataset_probs 1 \
#      --flash_attn \
#      --gradient_checkpointing \
#      --group_size 1 \
#      --value_head_dim 2 \
#      --save_best_model 2 \
#      --train_split_ratio 1 \
#      --cache_dir /scr/shijie/hf/base \
#      --freeze_vision_model \
#      --lora_rank 128 \
#      --lora_alpha 256 \
#      --ft_projector \
#      --add_cross_attention \
#      --is_general_preference \
#      --lora_dropout 0.05

# # phi3 cls lora
# deepspeed train_llava_reward.py \
#      --save_path /scr/shijie/hf/rewardmodel/test/phi3_test_cls_lora \
#      --save_steps 2 \
#      --logging_steps 1 \
#      --eval_steps 10000 \
#      --accumulated_gradient 4 \
#      --micro_train_batch_size 4 \
#      --pretrain microsoft/Phi-3.5-vision-instruct \
#      --bf16 \
#      --max_epochs 3 \
#      --max_len 2048 \
#      --zero_stage 3 \
#      --learning_rate 2e-4 \
#      --general_preference_tau 0.1 \
#      --dataset data/Unsafebench/Unsafebench_cap_train.json \
#      --dataset_probs 1 \
#      --flash_attn \
#      --gradient_checkpointing \
#      --group_size 1 \
#      --value_head_dim 1 \
#      --save_best_model 2 \
#      --train_split_ratio 1 \
#      --cache_dir /scr/shijie/hf/base \
#      --freeze_vision_model \
#      --lora_rank 128 \
#      --lora_alpha 256 \
#      --ft_projector \
#      --add_cross_attention \
#      --cls_based \
#      --lora_dropout 0.05
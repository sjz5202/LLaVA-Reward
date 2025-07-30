#llava lora bt
deepspeed train_llava_reward.py \
     --save_path /scr/shijie/hf/rewardmodel/test/llava_test_bt_lora \
     --save_steps 2 \
     --logging_steps 1 \
     --eval_steps 1000 \
     --accumulated_gradient 1 \
     --micro_train_batch_size 1 \
     --pretrain llava-hf/llava-v1.6-vicuna-13b-hf \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 2 \
     --learning_rate 2e-4 \
     --general_preference_tau 0.1 \
     --dataset data/ImageReward/train_data/imagereward_fidelity_rating_train_jpg.json \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --group_size 1 \
     --value_head_dim 1 \
     --save_best_model 2 \
     --train_split_ratio 1 \
     --freeze_vision_model \
     --cache_dir /scr/shijie/hf/base \
     --lora_rank 128 \
     --lora_alpha 256 \
     --ft_projector \
     --lora_dropout 0.05

#llava lora gpm
deepspeed train_llava_reward.py \
     --save_path /scr/shijie/hf/rewardmodel/test/llava_test_gpm_lora \
     --save_steps 2 \
     --logging_steps 1 \
     --eval_steps 1000 \
     --accumulated_gradient 1 \
     --micro_train_batch_size 1 \
     --pretrain llava-hf/llava-v1.6-vicuna-13b-hf \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 2 \
     --learning_rate 2e-4 \
     --general_preference_tau 0.1 \
     --dataset data/ImageReward/train_data/imagereward_fidelity_rating_train_jpg.json \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --group_size 1 \
     --value_head_dim 2 \
     --save_best_model 2 \
     --train_split_ratio 1 \
     --freeze_vision_model \
     --cache_dir /scr/shijie/hf/base \
     --lora_rank 128 \
     --lora_alpha 256 \
     --ft_projector \
     --is_general_preference  \
     --lora_dropout 0.05

#llava lora cls
deepspeed train_llava_reward.py \
     --save_path /scr/shijie/hf/rewardmodel/test/llava_test_cls_lora \
     --save_steps 2 \
     --logging_steps 1 \
     --eval_steps 1000 \
     --accumulated_gradient 1 \
     --micro_train_batch_size 1 \
     --pretrain llava-hf/llava-v1.6-vicuna-13b-hf \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 2 \
     --learning_rate 2e-4 \
     --general_preference_tau 0.1 \
     --dataset data/Unsafebench/Unsafebench_cap_train.json \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --group_size 1 \
     --value_head_dim 1 \
     --save_best_model 2 \
     --train_split_ratio 1 \
     --freeze_vision_model \
     --cache_dir /scr/shijie/hf/base \
     --lora_rank 128 \
     --lora_alpha 256 \
     --ft_projector \
     --cls_based \
     --lora_dropout 0.05



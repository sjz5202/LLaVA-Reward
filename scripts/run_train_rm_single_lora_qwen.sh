# qwen bt lora
deepspeed train_llava_reward.py \
     --save_path /scr/shijie/hf/rewardmodel/test/qwen_test_bt_lora  \
     --save_steps 2 \
     --logging_steps 1 \
     --eval_steps 1000 \
     --accumulated_gradient 4 \
     --micro_train_batch_size 4 \
     --pretrain Qwen/Qwen2.5-VL-7B-Instruct \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 2e-4 \
     --general_preference_tau 0.1 \
     --dataset data/open_preference/test_rest.json \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --group_size 1 \
     --value_head_dim 1 \
     --save_best_model 2 \
     --train_split_ratio 1 \
     --cache_dir /data/shijie/hf \
     --freeze_vision_model \
     --lora_rank 128 \
     --lora_alpha 256 \
     --ft_projector \
     --lora_dropout 0.05

# qwen gpm lora
deepspeed train_llava_reward.py \
     --save_path /scr/shijie/hf/rewardmodel/test/qwen_test_gpm_lora  \
     --save_steps 2 \
     --logging_steps 1 \
     --eval_steps 1000 \
     --accumulated_gradient 4 \
     --micro_train_batch_size 4 \
     --pretrain Qwen/Qwen2.5-VL-7B-Instruct \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 2e-4 \
     --general_preference_tau 0.1 \
     --dataset data/open_preference/test_rest.json \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --group_size 1 \
     --value_head_dim 2 \
     --save_best_model 2 \
     --train_split_ratio 1 \
     --cache_dir /data/shijie/hf \
     --freeze_vision_model \
     --lora_rank 128 \
     --lora_alpha 256 \
     --ft_projector \
     --is_general_preference \
     --lora_dropout 0.05

# qwen cls lora
deepspeed train_llava_reward.py \
     --save_path /scr/shijie/hf/rewardmodel/test/qwen_test_cls_lora  \
     --save_steps 2 \
     --logging_steps 1 \
     --eval_steps 1000 \
     --accumulated_gradient 4 \
     --micro_train_batch_size 4 \
     --pretrain Qwen/Qwen2.5-VL-7B-Instruct \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
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
     --cache_dir /data/shijie/hf \
     --freeze_vision_model \
     --lora_rank 128 \
     --lora_alpha 256 \
     --lora_dropout 0.05 \
     --ft_projector \
     --cls_based 



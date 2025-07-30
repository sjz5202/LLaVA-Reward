export PYTHONPATH=$(pwd)

python eval/batch_inference_rm_ft_phi.py \
--pm_path /scr/shijie/hf/rewardmodel/test/phi_test_gpm_ft/epoch_0_global_step_1 \
--dataset data/MJBench/mjbench_quality.json \
--max_samples 1000000 \
--general_preference_tau 0.1 \
--micro_batch_size 1 \
--value_head_dim 2 \
--is_custom_dataset \
--is_general_preference \
--add_cross_attention \
--device cuda:0
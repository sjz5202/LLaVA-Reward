export PYTHONPATH=$(pwd)

# replace your dataset with the similar format as the pairwise_test.json or non_pairwise_test.json in sample folder
# for non-pairwise dataset, add --cls_based, if you want to evaluate the classification probability
python eval/batch_inference_rm_qwen.py \
--pretrain Qwen/Qwen2.5-VL-7B-Instruct \
--pm_path model-path \
--dataset data/sample_test/pairwise_sample.json \
--max_samples 500000 \
--micro_batch_size 1 \
--is_custom_dataset 
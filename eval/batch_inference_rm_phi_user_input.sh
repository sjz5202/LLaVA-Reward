export PYTHONPATH=$(pwd)

# only pairwise preference 
python eval/batch_inference_rm_phi_user_input.py \
--pretrain microsoft/Phi-3.5-vision-instruct \
--pm_path /code/alignment/alignment_aug_gpm_1_crossatt_1_lora_1_proj_1_mean_0_batch_4/epoch_0/ \
--input_caption '[["A curious cat exploring a haunted mansion"],["A close-up photograph of a fat orange cat with lasagna in its mouth. Shot on Leica M6."]]' \
--input_imgs '[["data/sample_test/sample_img/0_c.jpg","data/sample_test/sample_img/0_r.jpg"],["data/sample_test/sample_img/1_c.jpg","data/sample_test/sample_img/1_r.jpg"]]' \
--max_samples 500000 \
--micro_batch_size 3 \
--is_custom_dataset 
# pairwise preference and reward scoring
python eval/batch_inference_rm_phi_user_input.py \
--pretrain microsoft/Phi-3.5-vision-instruct \
--pm_path /code/alignment/alignment_aug_bt_1_crossatt_1_lora_1_proj_1_mean_0_batch_4/epoch_0/  \
--input_caption '[["A curious cat exploring a haunted mansion"],["A close-up photograph of a fat orange cat with lasagna in its mouth. Shot on Leica M6."]]' \
--input_imgs '[["data/sample_test/sample_img/0_c.jpg","data/sample_test/sample_img/0_r.jpg"],["data/sample_test/sample_img/1_c.jpg","data/sample_test/sample_img/1_r.jpg"]]' \
--max_samples 500000 \
--micro_batch_size 3 \
--is_custom_dataset 
# non-pairwise data, reward scoring only
# for binary classification, add --cls_based
python eval/batch_inference_rm_phi_user_input.py \
--pretrain microsoft/Phi-3.5-vision-instruct \
--pm_path /code/alignment/alignment_aug_bt_1_crossatt_1_lora_1_proj_1_mean_0_batch_4/epoch_0/ \
--input_caption '[["A curious cat exploring a haunted mansion"],["A close-up photograph of a fat orange cat with lasagna in its mouth. Shot on Leica M6."]]' \
--input_imgs '[["data/sample_test/sample_img/0_c.jpg"],["data/sample_test/sample_img/1_c.jpg"]]' \
--max_samples 500000 \
--micro_batch_size 3 \
--is_custom_dataset
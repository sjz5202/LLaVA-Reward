import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from llava_reward.datasets import GeneralRewardDataset
from llava_reward.utils import blending_datasets, get_strategy, get_tokenizer,Qwen2RMSNorm,Phi3RMSNorm,LlamaRMSNorm
from llava_reward.models import get_reward_model,_get_reward_model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from llava_reward.models.base_mllm.phi3_v.modeling_phi3_v import Phi3VModel,Phi3VForCausalLM
from eval.reward_adaptor_loader import load_reward_adaptor
import os
import deepspeed
import time
import json
from scipy.stats import spearmanr, pearsonr
import yaml

def batch_rm_inference(args):
    
    class Empty:
        pass

    strategy = Empty()
    strategy.print = print
    strategy.is_rank_0 = lambda: True
    # prepare models
    args,model = load_reward_adaptor(args,model_type='phi3v',reward_config_path=os.path.join(args.pm_path, "reward_config.yaml"))
    strategy.args = args
    model.to('cuda')
    model.eval()
    processor,tokenizer = get_tokenizer(args.pretrain, model, "left", strategy=None,cache_dir= args.cache_dir, use_fast=not args.disable_fast_tokenizer)
    tokenizer.truncation_side = "right"

    if args.input_caption!=None and args.input_imgs!=None:
        dataset=[]
        args.input_caption=np.array(json.loads(args.input_caption))
        args.input_imgs=np.array(json.loads(args.input_imgs))
        if args.input_caption.shape[0]!=args.input_imgs.shape[0]:
            raise ValueError("The number of captions and images must be the same")
        if args.input_imgs.shape[1]==2:
            print('==================')
            print('Pairwise preference mode is used.')
            for i in range(args.input_caption.shape[0]):
                caption=args.input_caption[i][0]
                img_path_0=args.input_imgs[i][0]
                img_path_1=args.input_imgs[i][1]
                dataset.append({'prompt_id':i,'prompt':caption,'chosen_path':img_path_0,'c_rate':0,'reject_path':img_path_1,'r_rate':0})
            dataset = GeneralRewardDataset(dataset,processor=processor,tokenizer=tokenizer, strategy=strategy, is_custom=args.is_custom_dataset, return_prompt_length=False)
            sampler = DistributedSampler(
                dataset,
                num_replicas=1,
                rank=0,
                shuffle=False,
                seed=args.seed,
                drop_last=False,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=args.micro_batch_size,
                sampler=sampler,
                drop_last=False,
                collate_fn=dataset.collate_fn,
                pin_memory=False,
            )

            pbar = tqdm(
                dataloader,
                disable=not strategy.is_rank_0(),
            )                        
            all_probs = []
            execution_times=[]
            reward_list = []
            ground_truth_reward_list = []
            chosen_rewards_list=[]
            reject_rewards_list=[]
            with torch.no_grad():
                i=0
                for inputs_batch_c, inputs_batch_r,c_rates,r_rates in pbar:
                    chosen_ids = inputs_batch_c['input_ids'].squeeze(1).to(torch.cuda.current_device())
                    c_mask = inputs_batch_c['attention_mask'].squeeze(1).to(torch.cuda.current_device())
                    c_pixel_value=inputs_batch_c['pixel_values'].squeeze(1).to(torch.cuda.current_device())
                    c_img_size=inputs_batch_c['image_sizes'].squeeze(1).to(torch.cuda.current_device())

                    reject_ids = inputs_batch_r['input_ids'].squeeze(1).to(torch.cuda.current_device())
                    r_mask = inputs_batch_r['attention_mask'].squeeze(1).to(torch.cuda.current_device())
                    r_pixel_value=inputs_batch_r['pixel_values'].squeeze(1).to(torch.cuda.current_device())
                    r_img_size=inputs_batch_r['image_sizes'].squeeze(1).to(torch.cuda.current_device())
                    
                    start_time = time.time()
                    chosen_rewards, _ = model.custom_forward(chosen_ids, c_mask,c_pixel_value, c_img_size)
                    reject_rewards, _ = model.custom_forward(reject_ids, r_mask, r_pixel_value,r_img_size)
                    if not args.is_general_preference:
                        chosen_rewards_list.extend(chosen_rewards.squeeze(-1).tolist())
                        reject_rewards_list.extend(reject_rewards.squeeze(-1).tolist())
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                    average_time = sum(execution_times) / len(execution_times)
                    # print(f"Average execution time of model.custom_forward: {average_time/2} seconds")
                    if args.is_general_preference and args.value_head_dim == 2:
                        gpm_product = chosen_rewards[:, 0] * reject_rewards[:, 1] - chosen_rewards[:, 1] * reject_rewards[:, 0]
                        prob = F.sigmoid(gpm_product/args.general_preference_tau)
                    else:
                        prob = F.sigmoid((chosen_rewards - reject_rewards)/args.general_preference_tau).squeeze(-1)
                    prob = prob.float().cpu().numpy()
                    prob_array = [prob[index] for index in range(prob.shape[0])]
                    all_probs.extend(prob_array)
                    greater_than_half = [x for x in all_probs if x > 0.5]
                    count_greater_than_half = len(greater_than_half)
                    total_count = len(all_probs)
                    proportion = count_greater_than_half / total_count
            if not args.is_general_preference:
                print('image0 reward:',chosen_rewards_list) 
                print('image1 reward:',reject_rewards_list)
            print('Predict probability that image0 is better than image1:',all_probs)
        elif args.input_imgs.shape[1]==1:
            if args.is_general_preference:
                raise ValueError("General preference loss-based model is not supported for single image evaluation. Please use BT model instead.")
            print('==================') 
            print('Single image evaluation mode is used.')
            for i in range(args.input_caption.shape[0]):
                caption=args.input_caption[i]
                img_path=args.input_imgs[i][0]
                dataset.append({'path':img_path,'label':0,'prompt':caption})
            dataset = GeneralRewardDataset(dataset,processor=processor,tokenizer=tokenizer, strategy=strategy, is_custom=args.is_custom_dataset, return_prompt_length=False,cls_based=True)
            sampler = DistributedSampler(
                dataset,
                num_replicas=1,
                rank=0,
                shuffle=False,
                seed=args.seed,
                drop_last=False,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=args.micro_batch_size,
                sampler=sampler,
                drop_last=False,
                collate_fn=dataset.collate_fn,
                pin_memory=False,
            )

            pbar = tqdm(
                dataloader,
                disable=not strategy.is_rank_0(),
            )
            reward_list=[]
            prob_list=[]
            with torch.no_grad():
                for input_batch, labels in pbar:
                    ids = input_batch['input_ids'].squeeze(1).to(torch.cuda.current_device())
                    masks = input_batch['attention_mask'].squeeze(1).to(torch.cuda.current_device())
                    pixel_values=input_batch['pixel_values'].squeeze(1).to(torch.cuda.current_device())
                    img_sizes=input_batch['image_sizes'].squeeze(1).to(torch.cuda.current_device())
                    labels=labels.to(torch.cuda.current_device())
                    rewards, _ = model.custom_forward(input_ids=ids, attention_mask=masks,pixel_values=pixel_values ,image_sizes=img_sizes)
                    if not args.is_general_preference:
                        reward_list.extend(rewards.squeeze(-1).tolist())
                    if args.cls_based:
                        prob_list.extend(F.sigmoid(rewards).tolist())
            print('Reward:',reward_list)
            if args.cls_based:
                print('Classification probability:',prob_list)
    else:
        raise ValueError("Please provide input_caption and input_imgs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm_path", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--add_cross_attention", action="store_true", default=False, help="add_cross_attention")
    parser.add_argument("--mean_hidden_state", action="store_true", default=False, help="mean_hidden_state")
    parser.add_argument("--layer_id", type=int, default=32)
    parser.add_argument("--input_caption", type=str, default=None)
    parser.add_argument("--input_imgs", type=str, default=None)
    parser.add_argument("--input_label", type=str, default=None)
    parser.add_argument("--cls_based", action="store_true", default=False, help="Whether cls_based.")
    # custom arguments added 
    parser.add_argument("--is_custom_dataset", action="store_true", default=False, help="Whether to use custom cyclic dataset. Default to False.")
    parser.add_argument("--is_general_preference", action="store_true", default=False, help="Whether to use General Preference model. Default to False (Bradley Terry model by default).")
    parser.add_argument("--ft_projector", action="store_true", default=True, help="Whether to use General Preference model. Default to False (Bradley Terry model by default).")
    parser.add_argument("--general_preference_tau", type=float, default=0.1, help="Hyperparameter tau used in general preference loss.")
    parser.add_argument("--value_head_dim", type=int, default=1, help="Dimension of the value head in the general preference model. Ignored by the Bradley Terry model. Should be even.")


    args = parser.parse_args()
    
    batch_rm_inference(args)
    


import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from llava_reward.datasets import GeneralRewardDataset,GeneralRewardDataset_qwen
from llava_reward.utils import blending_datasets, get_strategy, get_tokenizer,get_tokenizer_qwen,get_tokenizer_llava,Qwen2RMSNorm,Phi3RMSNorm
from llava_reward.models import get_reward_model,_get_reward_model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  transformers import Qwen2_5_VLModel,Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from llava_reward.models.base_mllm.phi3_v.modeling_phi3_v import Phi3VModel,Phi3VForCausalLM
import os
import yaml
from sklearn.metrics import f1_score, recall_score
from eval.reward_adaptor_loader import load_reward_adaptor

def batch_rm_inference(args):
    
    class Empty:
        pass

    strategy = Empty()
    strategy.print = print
    strategy.is_rank_0 = lambda: True
    # prepare models
    args,model = load_reward_adaptor(args,model_type='qwen',reward_config_path=os.path.join(args.pm_path, "reward_config.yaml"))
    strategy.args = args
    processor,tokenizer = get_tokenizer_qwen(args.pretrain, model, "left", strategy, cache_dir=args.cache_dir, use_fast=not args.disable_fast_tokenizer)
    tokenizer.truncation_side = "right"
    model.to('cuda')
    model.eval()
    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if len(dataset[0]) == 3:
        non_pairwise=True
    else:
        non_pairwise=False
    dataset = GeneralRewardDataset_qwen(dataset,processor=processor,tokenizer=tokenizer, strategy=strategy, is_custom=args.is_custom_dataset, return_prompt_length=False,cls_based=non_pairwise)
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
    if not non_pairwise:
        print('==================')
        print('Pairwise preference mode is used.')
        all_probs = []
        chosen_rewards_list=[]
        reject_rewards_list=[]
        with torch.no_grad():
            for inputs_batch_c, inputs_batch_r,c_rates,r_rates in pbar:
                inputs_batch_c=inputs_batch_c.to(torch.cuda.current_device())
                inputs_batch_r=inputs_batch_r.to(torch.cuda.current_device())
                chosen_rewards, c_outputs = model.custom_forward(inputs_batch=inputs_batch_c)
                reject_rewards, c_outputs = model.custom_forward(inputs_batch=inputs_batch_r)
                if not args.is_general_preference:
                    chosen_rewards_list.extend(chosen_rewards.squeeze(-1).tolist())
                    reject_rewards_list.extend(reject_rewards.squeeze(-1).tolist())
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
        greater_than_half = [x for x in all_probs if x > 0.5]
        count_greater_than_half = len(greater_than_half)
        total_count = len(all_probs)
        proportion = count_greater_than_half / total_count
        half = [x for x in all_probs if x == 0.5]
        prob_mean = sum(all_probs)/len(all_probs)
        print("prob_mean", prob_mean)
        print("final proportion", proportion) 
        if total_count-len(half)!=0:
            print("final proportion wo tie", count_greater_than_half / (total_count-len(half))) 
    elif non_pairwise:
        if args.is_general_preference:
            raise ValueError("General preference loss-based model is not supported for single image evaluation. Please use BT model instead.")
        print('==================')
        print('Single image evaluation mode is used.')
        reward_list=[]
        prob_list=[]    
        label_list=[]
        with torch.no_grad():
            for inputs_batch,labels in pbar:
                inputs_batch=inputs_batch.to(torch.cuda.current_device())
                labels=labels.to(torch.cuda.current_device())
                rewards, _ = model.custom_forward(inputs_batch=inputs_batch)
                label_list.extend(labels.tolist())
                if not args.is_general_preference:
                    reward_list.extend(rewards.squeeze(-1).tolist())
                if args.cls_based:
                    prob_list.extend(F.sigmoid(rewards).tolist())
        print('Reward:',reward_list)
        if args.cls_based:
            prob_list = np.array(prob_list)
            prob_list[prob_list>=0.5]=1
            prob_list[prob_list<0.5]=0
            accuracy=sum(p == l for p, l in zip(prob_list, label_list))/len(label_list)
            f1 = f1_score(label_list, prob_list, average='binary')
            recall = recall_score(label_list, prob_list)
            print(f"Accuracy: {accuracy}, F1 Score: {f1}, recall: {recall}")

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
    parser.add_argument("--layer_id", type=int, default=32)
    # custom arguments added 
    parser.add_argument("--is_custom_dataset", action="store_true", default=False, help="Whether to use custom cyclic dataset. Default to False.")
    parser.add_argument("--is_general_preference", action="store_true", default=False, help="Whether to use General Preference model. Default to False (Bradley Terry model by default).")
    parser.add_argument("--ft_projector", action="store_true", default=True, help="Whether to use General Preference model. Default to False (Bradley Terry model by default).")
    parser.add_argument("--general_preference_tau", type=float, default=0.1, help="Hyperparameter tau used in general preference loss.")
    parser.add_argument("--value_head_dim", type=int, default=1, help="Dimension of the value head in the general preference model. Ignored by the Bradley Terry model. Should be even.")
    parser.add_argument("--cls_based", action="store_true", default=False, help="Whether cls_based.")

    args = parser.parse_args()
    
    batch_rm_inference(args)
    


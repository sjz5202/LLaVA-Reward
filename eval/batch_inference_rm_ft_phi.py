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
from  transformers import Qwen2_5_VLModel,Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from llava_reward.models.base_mllm.phi3_v.modeling_phi3_v import Phi3VModel,Phi3VForCausalLM
import os

def batch_rm_inference(args):
    
    class Empty:
        pass

    strategy = Empty()
    strategy.print = print
    strategy.is_rank_0 = lambda: True
    strategy.args = args

    device = torch.device(args.device)
    if args.add_cross_attention:
        if 'qwen' in args.pm_path.lower():
            cls_class=_get_reward_model(Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLModel, is_general_preference=args.is_general_preference,add_cross_attention=args.add_cross_attention,
                                        value_head_dim=args.value_head_dim,
                                        RMSNorm_class=Qwen2RMSNorm,
                                        RMSNorm_class_eps=1e-6)
        else:
            cls_class=_get_reward_model(Phi3VForCausalLM, Phi3VModel, RMSNorm_class=Phi3RMSNorm,RMSNorm_class_eps=1e-5,is_general_preference=args.is_general_preference,add_cross_attention=args.add_cross_attention, add_prompt_head=False, value_head_dim=args.value_head_dim)
    else:
        if 'qwen' in args.pm_path.lower():
            cls_class=_get_reward_model(Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLModel, is_general_preference=args.is_general_preference,
                                        value_head_dim=args.value_head_dim,
                                        RMSNorm_class=Qwen2RMSNorm,
                                        RMSNorm_class_eps=1e-6)
        else:
            cls_class=_get_reward_model(Phi3VForCausalLM, Phi3VModel, is_general_preference=args.is_general_preference, add_prompt_head=False, value_head_dim=args.value_head_dim,RMSNorm_class=Phi3RMSNorm,RMSNorm_class_eps=1e-5)
    config = AutoConfig.from_pretrained(args.pm_path, trust_remote_code=True,cache_dir= args.cache_dir)
    model = cls_class.from_pretrained(
            args.pm_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir= args.pm_path,
        )
    model.model_type='phi3v'
    processor,tokenizer = get_tokenizer(args.pm_path, model, "left", strategy=None,cache_dir= args.cache_dir, use_fast=not args.disable_fast_tokenizer)
    tokenizer.truncation_side = "right"
    model.to(device)
    # prepare models
    model.eval()
    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
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
    
    with torch.no_grad():
        for inputs_batch_c, inputs_batch_r,c_rates,r_rates in pbar:
            chosen_ids = inputs_batch_c['input_ids'].squeeze(1).to(torch.cuda.current_device())
            c_mask = inputs_batch_c['attention_mask'].squeeze(1).to(torch.cuda.current_device())
            c_pixel_value=inputs_batch_c['pixel_values'].squeeze(1).to(torch.cuda.current_device())
            c_img_size=inputs_batch_c['image_sizes'].squeeze(1).to(torch.cuda.current_device())

            reject_ids = inputs_batch_r['input_ids'].squeeze(1).to(torch.cuda.current_device())
            r_mask = inputs_batch_r['attention_mask'].squeeze(1).to(torch.cuda.current_device())
            r_pixel_value=inputs_batch_r['pixel_values'].squeeze(1).to(torch.cuda.current_device())
            r_img_size=inputs_batch_r['image_sizes'].squeeze(1).to(torch.cuda.current_device())
            chosen_rewards, _ = model.custom_forward(input_ids=chosen_ids, attention_mask=c_mask,pixel_values=c_pixel_value,image_sizes=c_img_size)
            reject_rewards, _ = model.custom_forward(input_ids=reject_ids, attention_mask=r_mask, pixel_values=r_pixel_value,image_sizes=r_img_size)
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
            print('predict probability:',prob_array)
            print('groundtruth chosen img rating:',c_rates)
            print('groundtruth rejected img rating:',r_rates)
            print("proportion", proportion) 



    greater_than_half = [x for x in all_probs if x > 0.5]
    count_greater_than_half = len(greater_than_half)
    total_count = len(all_probs)
    proportion = count_greater_than_half / total_count
    
    prob_mean = sum(all_probs)/len(all_probs)
    print("prob_mean", prob_mean)
    print("final proportion", proportion) 


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
    
    # custom arguments added 
    parser.add_argument("--is_custom_dataset", action="store_true", default=False, help="Whether to use custom cyclic dataset. Default to False.")
    parser.add_argument("--is_general_preference", action="store_true", default=False, help="Whether to use General Preference model. Default to False (Bradley Terry model by default).")
    parser.add_argument("--general_preference_tau", type=float, default=0.1, help="Hyperparameter tau used in general preference loss.")
    parser.add_argument("--value_head_dim", type=int, default=2, help="Dimension of the value head in the general preference model. Ignored by the Bradley Terry model. Should be even.")
    parser.add_argument("--add_cross_attention", action="store_true", default=False, help="add_cross_attention")



    args = parser.parse_args()
    
    batch_rm_inference(args)
    


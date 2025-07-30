import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from llava_reward.datasets import GeneralRewardDataset
from llava_reward.utils import blending_datasets, get_strategy, get_tokenizer,Qwen2RMSNorm,Phi3RMSNorm,LlamaRMSNorm
from llava_reward.models import get_reward_model,_get_reward_model, PairWiseLoss, GeneralPreferenceLoss
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from llava_reward.models.base_mllm.phi3_v.modeling_phi3_v import Phi3VModel,Phi3VForCausalLM
import os
import deepspeed
import time
import json
from scipy.stats import spearmanr, pearsonr
import yaml
from  transformers import Qwen2_5_VLModel,Qwen2_5_VLForConditionalGeneration,LlavaNextForConditionalGeneration
from llava_reward.utils import get_tokenizer, get_tokenizer_qwen, get_tokenizer_llava
from PIL import Image

def load_reward_adaptor(args,model_type,reward_config_path,load_tokenizer=False):
    with open(reward_config_path) as f:
        reward_cfg = yaml.safe_load(f)
    args.is_general_preference=reward_cfg['is_general_preference']
    args.add_cross_attention=reward_cfg['add_cross_attention']
    args.value_head_dim=reward_cfg['value_head_dim']
    args.general_preference_tau=reward_cfg['general_preference_tau']
    if model_type=='phi3v':
        cls_class=_get_reward_model(Phi3VForCausalLM, Phi3VModel,RMSNorm_class=Phi3RMSNorm,RMSNorm_class_eps=1e-5, is_general_preference=args.is_general_preference,add_cross_attention=args.add_cross_attention, value_head_dim=args.value_head_dim)
        config = AutoConfig.from_pretrained(args.pretrain, trust_remote_code=True,cache_dir= args.cache_dir)
        model = cls_class.from_pretrained(
                args.pretrain,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir= args.cache_dir,
            )
        model.model_type='phi3v'
        from llava_reward.utils.utils import create_lora_config, patch_clip_for_lora
        patch_clip_for_lora(model)
        model.load_adapter(os.path.join(args.pm_path, "lora"))
        model.enable_adapters()
        state_dict = torch.load(os.path.join(args.pm_path, "pytorch_model.bin"))
        value_head_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'value_head' in k}
        model.value_head.load_state_dict(value_head_keys)
        if args.add_cross_attention:
            W_q_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'W_q' in k}
            model.W_q.load_state_dict(W_q_keys)
            W_k_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'W_k' in k}
            model.W_k.load_state_dict(W_k_keys)
            W_v_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'W_v' in k}
            model.W_v.load_state_dict(W_v_keys)
            ca_layernorm_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'ca_layernorm' in k}
            model.ca_layernorm.load_state_dict(ca_layernorm_keys)
        if args.ft_projector:
            img_projection_value_head_keys = {'.'.join(k.split('.')[-2:]): v for k, v in state_dict.items() if 'img_projection' in k}
            model.model.vision_embed_tokens.img_projection.load_state_dict(img_projection_value_head_keys)
        if load_tokenizer:
            processor, tokenizer = get_tokenizer(args.pretrain, model, "left", strategy=None, cache_dir=args.cache_dir, use_fast=not args.disable_fast_tokenizer)
            tokenizer.truncation_side = "right"
    elif model_type=='qwen':
        cls_class=_get_reward_model(Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLModel, is_general_preference=args.is_general_preference,add_cross_attention=args.add_cross_attention,
                                value_head_dim=args.value_head_dim,
                                RMSNorm_class=Qwen2RMSNorm,
                                RMSNorm_class_eps=1e-6)
        config = AutoConfig.from_pretrained(args.pretrain, trust_remote_code=True,cache_dir= args.cache_dir)
        model = cls_class.from_pretrained(
                args.pretrain,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir= args.cache_dir,
                attn_implementation="flash_attention_2"
            )
        model.model_type='qwen'
        from llava_reward.utils.utils import create_lora_config, patch_clip_for_lora
        model.load_adapter(os.path.join(args.pm_path, "lora"))
        model.enable_adapters()
        state_dict = torch.load(os.path.join(args.pm_path, "pytorch_model.bin"))
        value_head_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'value_head' in k}
        model.value_head.load_state_dict(value_head_keys)
        if args.add_cross_attention:
            print('load add_cross_attention')
            W_q_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'W_q' in k}
            model.W_q.load_state_dict(W_q_keys)
            W_k_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'W_k' in k}
            model.W_k.load_state_dict(W_k_keys)
            W_v_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'W_v' in k}
            model.W_v.load_state_dict(W_v_keys)
            ca_layernorm_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'ca_layernorm' in k}
            model.ca_layernorm.load_state_dict(ca_layernorm_keys)
        if args.ft_projector:
            print('qwen load ft_projector')
            img_projection_value_head_keys = {'.'.join(k.split('.')[-2:]): v for k, v in state_dict.items() if 'merger' in k}
            new_img_projection_value_head_keys = {
                "ln_q.weight": img_projection_value_head_keys["ln_q.weight"],
                "mlp.0.weight": img_projection_value_head_keys["0.weight"],
                "mlp.0.bias": img_projection_value_head_keys["0.bias"],
                "mlp.2.weight": img_projection_value_head_keys["2.weight"],
                "mlp.2.bias": img_projection_value_head_keys["2.bias"]
            }
            # print(new_img_projection_value_head_keys)
            model.visual.merger.load_state_dict(new_img_projection_value_head_keys)
        if load_tokenizer:
            processor, tokenizer = get_tokenizer_qwen(args.pretrain, model, "left", strategy=None, cache_dir=args.cache_dir, use_fast=not args.disable_fast_tokenizer)
            tokenizer.truncation_side = "right"
    elif model_type=='llava':
        cls_class=_get_reward_model(LlavaNextForConditionalGeneration,LlavaNextForConditionalGeneration, is_general_preference=args.is_general_preference,add_cross_attention=args.add_cross_attention,
                                value_head_dim=args.value_head_dim,
                                RMSNorm_class=LlamaRMSNorm,
                                RMSNorm_class_eps=1e-5,
                                model_type='llava')
        config = AutoConfig.from_pretrained(args.pretrain, trust_remote_code=True,cache_dir= args.cache_dir)
        model = cls_class.from_pretrained(
                args.pretrain,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir= args.cache_dir,
            )
        from llava_reward.utils.utils import create_lora_config, patch_clip_for_lora
        model.load_adapter(os.path.join(args.pm_path, "lora"))
        model.enable_adapters()
        state_dict = torch.load(os.path.join(args.pm_path, "pytorch_model.bin"))
        value_head_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'value_head' in k}
        model.value_head.load_state_dict(value_head_keys)
        if args.add_cross_attention:
            print('load add_cross_attention')
            W_q_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'W_q' in k}
            model.W_q.load_state_dict(W_q_keys)
            W_k_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'W_k' in k}
            model.W_k.load_state_dict(W_k_keys)
            W_v_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'W_v' in k}
            model.W_v.load_state_dict(W_v_keys)
            ca_layernorm_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'ca_layernorm' in k}
            model.ca_layernorm.load_state_dict(ca_layernorm_keys)
        if args.ft_projector:
            print('llava load ft_projector')
            img_projection_value_head_keys = {'.'.join(k.split('.')[-2:]): v for k, v in state_dict.items() if 'multi_modal_projector' in k}
            new_img_projection_value_head_keys = {
                "linear_1.weight": img_projection_value_head_keys["linear_1.weight"],
                "linear_1.bias": img_projection_value_head_keys["linear_1.bias"],
                "linear_2.weight": img_projection_value_head_keys["linear_2.weight"],
                "linear_2.bias": img_projection_value_head_keys["linear_2.bias"],
            }
            model.multi_modal_projector.load_state_dict(new_img_projection_value_head_keys)
        if load_tokenizer:
            processor, tokenizer = get_tokenizer_llava(args.pretrain, model, "left", strategy=None, cache_dir=args.cache_dir, use_fast=not args.disable_fast_tokenizer)
            tokenizer.truncation_side = "right"
    if load_tokenizer:
        return args,model,processor,tokenizer
    else:
        return args,model

def inference_process_phi3v(args,processor,tokenizer,img_dir_list,caption,device='cuda'):
    img_list = []
    img_inputs = []
    for dir in img_dir_list:
        img_list.append(Image.open(dir).convert("RGB"))
    prompt_messages = {
        'role': 'user',
        'content': f'<|image_1|>\n{caption}',
    }
    prompt = tokenizer.apply_chat_template([prompt_messages], tokenize=False, add_generation_prompt=True)[:-22]+tokenizer.eos_token
    for img in img_list:
        img_input=processor(text=prompt, images=[img], return_tensors="pt", padding=True, truncation=True)
        for k in img_input:
            img_input[k] = img_input[k].to(device)
        img_inputs.append(img_input)
    return img_inputs
def preference_compute(args,chosen_rewards,reject_rewards):
    if args.is_general_preference and args.value_head_dim == 2:
        gpm_product = chosen_rewards[:, 0] * reject_rewards[:, 1] - chosen_rewards[:, 1] * reject_rewards[:, 0]
        prob = F.sigmoid(gpm_product/args.general_preference_tau)
    else:
        prob = F.sigmoid((chosen_rewards - reject_rewards)/args.general_preference_tau).squeeze(-1)
    prob = prob.float().cpu().numpy()
    return prob
# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py

from typing import Optional
import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from llava_reward.utils.logging import init_logger
import torch.nn.functional as F
from llava_reward.models.base_mllm.phi3_v.modeling_phi3_v import Phi3VModel,Phi3VForCausalLM
from transformers import Qwen2_5_VLModel,Qwen2_5_VLForConditionalGeneration,LlavaNextForConditionalGeneration
from llava_reward.utils.utils import create_lora_config,create_lora_config_qwen,patch_clip_for_lora,create_lora_config_llava16_vicuna
import os
import math
logger = init_logger(__name__)
class Phi3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
# Construct reward model with a value head for sequence classification. (model also with a lm head) 
def get_reward_model(
    model_name_or_path: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    init_prompt_head: bool = False,
    add_prompt_head: bool = False,
    is_general_preference: bool = False,
    mean_hidden_state: bool=False,
    add_cross_attention: bool = False,
    value_head_dim: int = 2,
    cache_dir: str = None,
    freeze_vision_model: bool = False,
    is_pretrained_pm: str = None, 
    ft_projector: bool = False,
    layer_id: int = 32,
    **kwargs,
) -> nn.Module:
    """Get reward model with a value head(linear layer) and a lm head.

    Args:
        model_name_or_path (str): Path to pretrained model.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.
        init_value_head (bool, optional): Whether to initialize the value head weights. Defaults to False.
        is_general_preference (bool, optional): Whether to use General Preference model. Defaults to False (Bradley Terry model by default).
        value_head_dim (int, optional): Dimension of value head for General Prefernce model. Ignored by the Bradley Terry model. Defaults to 2.

    Returns:
        nn.Module: pretrained transformer model.
    """

    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir,trust_remote_code=True)
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    if 'phi' in model_name_or_path.lower():
        model_type = 'phi3v'
        base_class =Phi3VModel
        base_causal_class = Phi3VForCausalLM
        RMSNorm_class=Phi3RMSNorm
        RMSNorm_class_eps=1e-5
        lora_config = create_lora_config(
            rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_vision_model=freeze_vision_model,
        )
    elif 'qwen' in model_name_or_path.lower():
        model_type = 'qwen'
        base_class =Qwen2_5_VLModel  
        base_causal_class = Qwen2_5_VLForConditionalGeneration
        RMSNorm_class=Qwen2RMSNorm
        RMSNorm_class_eps=1e-6
        lora_config = create_lora_config_qwen(
            rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_vision_model=freeze_vision_model,
        )
    elif 'llava' in model_name_or_path.lower():
        model_type = 'llava'
        base_class =LlavaNextForConditionalGeneration  
        base_causal_class = LlavaNextForConditionalGeneration
        RMSNorm_class=LlamaRMSNorm
        RMSNorm_class_eps=1e-5
        lora_config = create_lora_config_llava16_vicuna(
            rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_vision_model=freeze_vision_model,
        )
    else:
        base_class =Phi3VModel
        base_causal_class = Phi3VForCausalLM
        RMSNorm_class=Phi3RMSNorm
        RMSNorm_class_eps=1e-5
        lora_config = create_lora_config(
            rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout=lora_dropout,
            freeze_vision_model=freeze_vision_model,
        )
    cls_class = _get_reward_model(base_causal_model=base_causal_class, base_llm_model=base_class, RMSNorm_class=RMSNorm_class,RMSNorm_class_eps=RMSNorm_class_eps,layer_id=layer_id,is_general_preference=is_general_preference, add_cross_attention=add_cross_attention,add_prompt_head=add_prompt_head, value_head_dim=value_head_dim,lora_config=lora_config,model_type=model_type,mean_hidden_state=mean_hidden_state)    
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        **kwargs,
    )
    # LoRA
    if lora_rank > 0:
        if 'phi' in model_name_or_path.lower():
            patch_clip_for_lora(model)
            # model = get_peft_model(model, lora_config)
            if is_pretrained_pm is not None:
                model.load_adapter(os.path.join(is_pretrained_pm, "lora"))
            else:
                model.add_adapter(lora_config)
            model.enable_adapters()
            if freeze_vision_model:
                model.model.vision_embed_tokens.requires_grad_(False)
            if ft_projector:
                model.model.vision_embed_tokens.img_projection.requires_grad_(True)
        elif 'qwen' in model_name_or_path.lower():
            if is_pretrained_pm is not None:
                model.load_adapter(os.path.join(is_pretrained_pm, "lora"))
            else:
                model.add_adapter(lora_config)
            model.enable_adapters()
            if freeze_vision_model:
                model.visual.requires_grad_(False)
            if ft_projector:
                model.visual.merger.requires_grad_(True)
        elif 'llava' in model_name_or_path.lower():
            if is_pretrained_pm is not None:
                model.load_adapter(os.path.join(is_pretrained_pm, "lora"))
            else:
                model.add_adapter(lora_config)
            model.enable_adapters()
            if freeze_vision_model:
                model.vision_tower.requires_grad_(False)
            if ft_projector:
                model.multi_modal_projector.requires_grad_(True)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module.to(torch.bfloat16)
                if "norm" in name:
                    module.to(torch.float32)
                if "value_head" in name or "embed_tokens" in name or "unpaired_value_head" in name:
                    if hasattr(module, "weight"):
                        module.to(torch.bfloat16)
    if 'llava' in model_name_or_path.lower():
        hidden_size=config.text_config.hidden_size
    else:
        hidden_size=config.hidden_size

    if init_value_head:
        if dschf is not None:
            logger.info("Initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
                    if is_pretrained_pm is not None:
                        state_dict = torch.load(os.path.join(is_pretrained_pm, "pytorch_model.bin"))
                        value_head_keys = {k.split('.')[-1]: v for k, v in state_dict.items() if 'value_head' in k}
                        model.value_head.load_state_dict(value_head_keys)
        else:
            model.value_head.weight.data.normal_(mean=0.0, std=1 / (hidden_size + 1))
            
    if init_prompt_head and add_prompt_head:
        if dschf is not None:
            logger.info("Initialize prompt_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.prompt_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.prompt_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.prompt_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    if lora_rank <= 0:
        if freeze_vision_model:
            if 'phi' in model_name_or_path.lower():
                model.model.vision_embed_tokens.requires_grad_(False)
                model.model.vision_embed_tokens.img_projection.requires_grad_(True)
            elif 'qwen' in model_name_or_path.lower():
                model.visual.requires_grad_(False)
                model.visual.merger.requires_grad_(True)
            elif 'llava' in model_name_or_path.lower():
                model.vision_tower.requires_grad_(False)
                model.multi_modal_projector.requires_grad_(True)

    model.value_head.requires_grad_(True)

    if hasattr(model, "W_k"):
        with deepspeed.zero.GatheredParameters([model.W_q.weight], modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                model.W_q.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        with deepspeed.zero.GatheredParameters([model.W_k.weight], modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                model.W_k.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        with deepspeed.zero.GatheredParameters([model.W_v.weight], modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                model.W_v.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        model.W_q.requires_grad_(True)
        model.W_k.requires_grad_(True)
        model.W_v.requires_grad_(True)
        model.ca_layernorm.requires_grad_(True)
    if add_prompt_head:
        model.prompt_head.requires_grad_(True)
        
    return model

def _get_reward_model(
    base_causal_model, 
    base_llm_model,
    RMSNorm_class,
    RMSNorm_class_eps,
    layer_id: int = 32, 
    vision_layer_id: int = -1,
    is_general_preference: bool=False, 
    add_cross_attention: bool=False,
    add_prompt_head: bool=False,
    value_head_dim: int=2,
    lora_config=None,
    model_type=None,
    mean_hidden_state=None):    
    class CustomRewardModel(base_causal_model):
        supports_gradient_checkpointing = True
        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.model_type=model_type
            if self.model_type =='llava':
                hidden_size=config.text_config.hidden_size
            else:
                setattr(self, self.base_model_prefix, base_llm_model(config))
                hidden_size=config.hidden_size
            if add_cross_attention:
                self.W_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.W_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.W_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.ca_layernorm=RMSNorm_class(hidden_size,eps=RMSNorm_class_eps)
            if not is_general_preference:
                self.value_head_dim=1
                self.value_head = nn.Linear(hidden_size, 1, bias=False)
            else: 
                self.value_head_dim=value_head_dim
                self.value_head = nn.Linear(hidden_size, value_head_dim, bias=False) 
                if add_prompt_head:
                    self.prompt_head = nn.Linear(config.hidden_size, value_head_dim//2, bias=False) 
            self.add_cross_attention = add_cross_attention
            self.is_general_preference = is_general_preference  
            self.mean_hidden_state=mean_hidden_state  
            self.lora_config=lora_config
            self.post_init()
            self.layer_id=layer_id
            self.vision_layer_id=vision_layer_id
        def custom_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[torch.LongTensor] = None,
            return_output=False,
            inputs_batch=None
        ) -> torch.Tensor:
            if self.model_type=='phi3v':
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids,pixel_values=pixel_values,image_sizes=image_sizes,output_hidden_states=True
                )
                if self.layer_id==32:
                    last_hidden_states = outputs["last_hidden_state"]
                else: 
                    last_hidden_states = outputs["hidden_states"][self.layer_id]
                vision_embedding=outputs["hidden_states"][self.vision_layer_id][:,:outputs['hidden_states'][-1].shape[1],:]
            elif self.model_type=='qwen':
                attention_mask=inputs_batch['attention_mask']
                vision_embedding=self.visual(inputs_batch['pixel_values'], grid_thw=inputs_batch['image_grid_thw'])
                outputs=self.forward(**inputs_batch,output_hidden_states=True)
                image_mask = (inputs_batch['input_ids'] == 151643)
                hidden_states=outputs["hidden_states"]
                last_hidden_states=outputs["hidden_states"][-1]
                vision_src=outputs["hidden_states"][0]
                B, L, H = last_hidden_states.shape
                vis_lens = image_mask.sum(dim=1)                # 每样本视觉 token 数
                max_v_len = int(vis_lens.max())
                vision_pad       = last_hidden_states.new_zeros(B, max_v_len, H)
                vision_pad_mask  = torch.ones(B, max_v_len, dtype=torch.bool,
                                            device=last_hidden_states.device)
                for i in range(B):
                    v_len = vis_lens[i]
                    vision_pad[i, :v_len]      = vision_src[i, image_mask[i]]
                    vision_pad_mask[i, :v_len] = False           # False = 有效位
            elif self.model_type=='llava':
                attention_mask=inputs_batch['attention_mask']
                outputs=self.forward(**inputs_batch,output_hidden_states=True)
                last_hidden_states=outputs["hidden_states"][-1]
            if self.add_cross_attention:
                if self.model_type=='phi3v':
                    Q_ = self.W_q(last_hidden_states) 
                    K_ = self.W_k(vision_embedding)     
                    V_ = self.W_v(vision_embedding) 
                    scores = torch.bmm(Q_, K_.transpose(1, 2)) 
                    d_k = vision_embedding.shape[2]                
                    scores = scores / math.sqrt(d_k)
                    attn_weights = F.softmax(scores, dim=-1)  
                    attn_output = torch.bmm(attn_weights, V_)
                    last_hidden_states=self.ca_layernorm(last_hidden_states + attn_output)
                elif self.model_type=='qwen':
                    Q_ = self.W_q(last_hidden_states)                     
                    K_ = self.W_k(vision_pad)                 
                    V_ = self.W_v(vision_pad)                   
                    scores = torch.bmm(Q_, K_.transpose(1, 2)) / math.sqrt(H)   # (B, L, V_max)
                    scores = scores.masked_fill(vision_pad_mask.unsqueeze(1), -1e4)
                    attn_w = F.softmax(scores, dim=-1)
                    attn_o = torch.bmm(attn_w, V_)                               # (B, L, H)
                    last_hidden_states=self.ca_layernorm(last_hidden_states + attn_o)
                    del vision_pad
                    torch.cuda.empty_cache()
            if self.mean_hidden_state:
                mask = attention_mask.to(dtype=last_hidden_states.dtype)
                if last_hidden_states.dim() == 3:
                    mask = mask.unsqueeze(-1)  
                sum_last_hidden_states = (last_hidden_states * mask).sum(dim=1)
                mask_lens = mask.sum(dim=1).clamp(min=1e-8)  
                last_hidden_states = sum_last_hidden_states / mask_lens
                if last_hidden_states.dim() == 3:
                    last_hidden_states = last_hidden_states.squeeze(-1)
            if not self.is_general_preference:
                values = self.value_head(last_hidden_states)
                # left padding in training mode
                if self.training:
                    if self.mean_hidden_state:
                        reward=values
                    else:
                        values=values.squeeze(-1)
                        reward = values[:, -1]
                else:
                    if self.mean_hidden_state:
                        reward=values
                    else:
                        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).unsqueeze(-1)
                        reward = values.gather(dim=1, index=eos_indices).squeeze(1)
                if return_output:
                    return reward, outputs
                else:
                    return reward, None
            else:
                values = self.value_head(last_hidden_states)
                # left padding in training mode
                if self.training:
                    # last token
                    if self.mean_hidden_state:
                        reward = values
                    else: 
                        reward = values[:, -1, :]
                else:
                    if self.mean_hidden_state:
                        reward = values
                    else:
                        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)
                        eos_indices = eos_indices.unsqueeze(1)  # [batch_size, 1]
                        reward_list = []
                        for dim in range(value_head_dim):
                            reward_list.append(values[:, :, dim].gather(dim=1, index=eos_indices))
                        reward = torch.cat(reward_list, dim=1)
                if return_output:
                    return reward, outputs
                else:
                    return reward, None
        
        def create_skew_symmetric_block_matrix(self, dim, device, dtype, prompt_hidden_states):
            """
            Create a batch of skew-symmetric block matrices where each matrix is data-dependent on
            the corresponding prompt_hidden_states. Only the relevant block diagonal parts are generated.
            
            Args:
            - dim: Dimension of the square matrix (must be even).
            - prompt_hidden_states: Tensor of shape [batch_size, hidden_dim].
            
            Returns:
            - batch_R_matrices: Tensor of shape [batch_size, dim, dim], with skew-symmetric block entries.
            """
            if hasattr(self, 'prompt_head'):
                batch_size = prompt_hidden_states.shape[0]
                
                # Ensure that dim is even, as we're creating blocks of size 2x2
                assert dim % 2 == 0, "dim must be even for skew-symmetric block generation"

                # Pass through the linear layer to get the block diagonal entries (half of the matrix's off-diagonal blocks)
                block_values = self.prompt_head(prompt_hidden_states).view(batch_size, dim // 2)
                block_values = torch.softmax(block_values, dim=-1)
                
                # Create a batch of zero matrices [batch_size, dim, dim]
                batch_R_matrices = torch.zeros((batch_size, dim, dim), device=device, dtype=dtype)
                
                # Fill only the block diagonal entries with the learned values
                for i in range(0, dim, 2):
                    batch_R_matrices[:, i, i + 1] = -block_values[:, i // 2]
                    batch_R_matrices[:, i + 1, i] = block_values[:, i // 2]  # Skew-symmetric condition
            else:
                raise AttributeError("prompt_head is not defined. Ensure 'add_prompt_head' is set to True during initialization.")
                
            return batch_R_matrices
        
                
    return CustomRewardModel







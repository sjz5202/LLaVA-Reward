# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/utils.py

import os
from pathlib import Path

from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer, AutoProcessor,LlavaNextProcessor
import torch
from peft import LoraConfig
from llava_reward.utils import DeepspeedStrategy
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, cache_dir=None,use_fast=True):
    processor = AutoProcessor.from_pretrained(pretrain,
                                            cache_dir=cache_dir, 
                                            padding_side=padding_side,
                                            trust_remote_code=True,
                                            num_crops=16,
                                            model_max_length=131072)
    tokenizer=processor.tokenizer
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return processor,tokenizer
    
def get_tokenizer_qwen(pretrain, model, padding_side="left", strategy=None, cache_dir=None,use_fast=True):
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(pretrain,cache_dir=cache_dir, min_pixels=min_pixels, max_pixels=max_pixels)
    tokenizer=processor.tokenizer
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return processor,tokenizer
    
def get_tokenizer_llava(pretrain, model, padding_side="left", strategy=None, cache_dir=None,use_fast=True):
    processor=LlavaNextProcessor.from_pretrained(pretrain,cache_dir=cache_dir, padding_side=padding_side)
    tokenizer=processor.tokenizer
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return processor,tokenizer

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

def get_strategy(args):
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    stopping_strategy="first_exhausted",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        dataset_subfold_list = dataset.split("@")
        strategy.print(f"dataset: {dataset}")
        # local dir with python script or common local file
        if os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                files = dataset
                data_type = os.path.splitext(files)[1][1:]
            else:
                path = Path(dataset)
                script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]
                strategy.print(f"script: {script}")
                strategy.print(f"files: {files}")
                # For dir, follow python script or first file type
                data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
            # reformat data type
            if data_type in ["json", "jsonl"]:
                data_type = "json"
            elif data_type == "txt":
                data_type = "text"
            elif data_type.endswith(".py"):
                # load local dir with python script
                files = None
            if data_type.endswith(".py"):
                strategy.print(f"load {dataset} with script {data_type}")
            else:
                strategy.print(f"load {files} from {dataset}")
            data = load_dataset(data_type, data_files=files)
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]
            subfold = dataset_subfold_list[1]
            data = load_dataset(dataset, data_dir=subfold.strip())
        elif len(dataset_subfold_list) == 1:
            dataset = dataset_subfold_list[0]
            data = load_dataset(dataset)
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")

        if "train" in data:
            train_data_list.append(data["train"].select(range(min(max_count, len(data["train"])))))
        else:
            train_data_list.append(data.select(range(min(max_count, len(data)))))  # train will contains eval? TODO
           
    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )

    return train_dataset

def create_lora_config(rank, lora_alpha=2.0, dropout=0.0, freeze_vision_model=False):
    linear_modules = [
        # Phi language modules
        'qkv_proj',  # attention
        'o_proj',
        'down_proj',  # MLP
        'gate_up_proj'
    ]
    if not freeze_vision_model:
        vision_linear_modules = [
            # CLIP modules
            'q_proj',  # attention
            'k_proj',
            'v_proj',
            'out_proj',
            'fc1',  # MLP
            'fc2',
            'img_projection.0',
            'img_projection.2',
        ]
        linear_modules.extend(vision_linear_modules)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=linear_modules,
        init_lora_weights='gaussian',
    )
    return lora_config
def create_lora_config_qwen(rank, lora_alpha=2.0, dropout=0.0, freeze_vision_model=False):
    linear_modules = [
        f"model.layers.{i}.self_attn.{proj}_proj"
        for i in range(28)  
        for proj in ["q", "k", "v", "o"]] + [
        f"model.layers.{i}.mlp.{proj}_proj"
        for i in range(28)
        for proj in ["gate", "up", "down"]
    ]
    if not freeze_vision_model:
        vision_linear_modules = []
        linear_modules.extend(vision_linear_modules)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=linear_modules,
        init_lora_weights='gaussian',
    )
    return lora_config
def create_lora_config_llava16_vicuna(rank, lora_alpha=2.0, dropout=0.0, freeze_vision_model=False):
    linear_modules = [
        f"language_model.model.layers.{i}.self_attn.{proj}_proj"
        for i in range(40)  
        for proj in ["q", "k", "v", "o"]] + [
        f"language_model.model.layers.{i}.mlp.{proj}_proj"
        for i in range(40)
        for proj in ["gate", "up", "down"]
    ]
    if not freeze_vision_model:
        vision_linear_modules = []
        linear_modules.extend(vision_linear_modules)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=linear_modules,
        init_lora_weights='gaussian',
    )
    return lora_config

def patch_clip_for_lora(model):
    # remove unused parameters and then monkey patch
    def get_img_features(self, img_embeds):
        clip_vision_model = self.img_processor.vision_model
        hidden_states = clip_vision_model.embeddings(img_embeds)
        hidden_states = clip_vision_model.pre_layrnorm(hidden_states)
        patch_feature = clip_vision_model.encoder(
            inputs_embeds=hidden_states, output_hidden_states=True
        ).hidden_states[-1][:, 1:]
        return patch_feature

    image_embedder = model.model.vision_embed_tokens
    layer_index = image_embedder.layer_idx
    clip_layers = image_embedder.img_processor.vision_model.encoder.layers
    if layer_index < 0:
        layer_index = len(clip_layers) + layer_index
    del clip_layers[layer_index + 1 :]
    del image_embedder.img_processor.vision_model.post_layernorm
    image_embedder.get_img_features = get_img_features.__get__(image_embedder)
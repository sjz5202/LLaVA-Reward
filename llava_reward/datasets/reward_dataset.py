from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences
from PIL import Image
from PIL import ImageFile
from llava_reward.models.base_mllm.qwen_vl_utils import process_vision_info

ImageFile.LOAD_TRUNCATED_IMAGES = True
def preprocess_data(data):
    prompt = data["prompt"]
    chosen = data["chosen_path"]
    reject = data["reject_path"]
    c_rate=data["c_rate"]
    r_rate=data["r_rate"]
    return prompt, chosen, reject,c_rate,r_rate
    
def preprocess_data_cls(data):
    prompt = data["prompt"]
    path = data["path"]
    label=data["label"]
    return prompt, path, label

class GeneralRewardDataset(Dataset):
    """
    General dataset for reward model, handling both custom and standard formats.

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        is_custom: flag indicating whether the dataset is in custom format
    """
    def __init__(
        self,
        dataset,
        processor: Callable,
        tokenizer: Callable,
        # max_length: int,
        strategy,
        is_custom=False,
        return_prompt_length=False,
        cls_based=False,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.tokenizer = tokenizer
        self.processor = processor
        self.strategy = strategy
        self.is_custom = is_custom
        self.return_prompt_length = return_prompt_length
        self.cls_based=cls_based
        if self.cls_based:
            self.path_list = []
            self.label_list = []
            for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
                prompt, path, label = preprocess_data_cls(data)
                self.prompts.append(prompt)
                self.path_list.append(path)
                self.label_list.append(label)
        else:
            self.chosens = []
            self.rejects = []
            self.c_rates = []
            self.r_rates = []
            for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
                prompt, chosen, reject,c_rate,r_rate = preprocess_data(data)
                self.prompts.append(prompt)
                self.chosens.append(chosen)
                self.rejects.append(reject)
                self.c_rates.append(c_rate)
                self.r_rates.append(r_rate)
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if not self.cls_based:
            prompt, chosen, reject,c_rate,r_rate= self.prompts[idx], self.chosens[idx], self.rejects[idx],self.c_rates[idx],self.r_rates[idx]
            chosen_img= Image.open(chosen).convert("RGB")
            reject_img= Image.open(reject).convert("RGB")
            if not isinstance(prompt, list):
                prompt_message = {
                    'role': 'user',
                    'content': f'<|image_1|>\n{prompt}',
                }
                prompt = self.tokenizer.apply_chat_template([prompt_message], tokenize=False, add_generation_prompt=True)[:-22]
                prompt=prompt+self.tokenizer.eos_token
                chosen_token=self.processor(prompt, [chosen_img], return_tensors='pt')
                reject_token=self.processor(prompt, [reject_img], return_tensors='pt')
            else: 
                prompt_message_c = {
                    'role': 'user',
                    'content': f'<|image_1|>\n{prompt[0]}',
                }
                prompt_c = self.tokenizer.apply_chat_template([prompt_message_c], tokenize=False, add_generation_prompt=True)[:-22]
                prompt_c=prompt_c+self.tokenizer.eos_token
                chosen_token=self.processor(prompt_c, [chosen_img], return_tensors='pt')

                prompt_message_r = {
                    'role': 'user',
                    'content': f'<|image_1|>\n{prompt[1]}',
                }
                prompt_r = self.tokenizer.apply_chat_template([prompt_message_r], tokenize=False, add_generation_prompt=True)[:-22]
                prompt_r=prompt_r+self.tokenizer.eos_token
                reject_token=self.processor(prompt_r, [reject_img], return_tensors='pt')
            return (
                chosen_token["input_ids"],
                chosen_token["attention_mask"],
                chosen_token["pixel_values"],
                chosen_token["image_sizes"],
                reject_token["input_ids"],
                reject_token["attention_mask"],
                reject_token["pixel_values"],
                reject_token["image_sizes"],
                c_rate,
                r_rate
            )
        else:
            prompt, path, label= self.prompts[idx], self.path_list[idx], self.label_list[idx]
            img= Image.open(path).convert("RGB")
            prompt_message = {
                'role': 'user',
                'content': f'<|image_1|>\n{prompt}',
            }
            prompt = self.tokenizer.apply_chat_template([prompt_message], tokenize=False, add_generation_prompt=True)[:-22]
            prompt=prompt+self.tokenizer.eos_token
            input_token=self.processor(prompt, [img], return_tensors='pt')
            return (
                input_token["input_ids"],
                input_token["attention_mask"],
                input_token["pixel_values"],
                input_token["image_sizes"],
                label
            )

    def collate_fn(self, item_list):
        if not self.cls_based:
            chosen_ids = []
            chosen_masks = []
            chosen_pixel_values=[]
            chosen_image_sizes=[]
            c_rate_list=[]

            reject_ids = []
            reject_masks = []
            reject_pixel_values=[]
            reject_image_sizes=[]
            r_rate_list=[]

            for chosen_id, chosen_mask,chosen_pixel_value, chosen_image_size, reject_id, reject_mask,reject_pixel_value,reject_image_size,c_rate,r_rate in item_list:
                chosen_ids.append(chosen_id)
                chosen_masks.append(chosen_mask)
                chosen_pixel_values.append(chosen_pixel_value)
                chosen_image_sizes.append(chosen_image_size)
                c_rate_list.append(c_rate)

                reject_ids.append(reject_id)
                reject_masks.append(reject_mask)
                reject_pixel_values.append(reject_pixel_value)
                reject_image_sizes.append(reject_image_size)
                r_rate_list.append(r_rate)
                
            chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
            chosen_masks = zero_pad_sequences(chosen_masks)
            reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
            reject_masks = zero_pad_sequences(reject_masks)
            inputs_batch_c = {
                "input_ids":      chosen_ids,
                "attention_mask": chosen_masks,
                "pixel_values":   torch.stack(chosen_pixel_values, dim=0),
                "image_sizes":    torch.stack(chosen_image_sizes, dim=0),
            }
            inputs_batch_r = {
                "input_ids":      reject_ids,
                "attention_mask": reject_masks,
                "pixel_values":   torch.stack(reject_pixel_values, dim=0),
                "image_sizes":    torch.stack(reject_image_sizes, dim=0),
            }
            return inputs_batch_c, inputs_batch_r,c_rate_list,r_rate_list
        else:
            ids = []
            masks = []
            pixel_values=[]
            image_sizes=[]
            label_list=[]

            for input_id, mask,pixel_value, image_size, label in item_list:
                ids.append(input_id)
                masks.append(mask)
                pixel_values.append(pixel_value)
                image_sizes.append(image_size)
                label_list.append(label)
            ids = zero_pad_sequences(ids, value=self.tokenizer.pad_token_id)
            masks = zero_pad_sequences(masks)
            inputs_batch = {
                "input_ids":      ids,
                "attention_mask": masks,
                "pixel_values":   torch.stack(pixel_values, dim=0),
                "image_sizes":    torch.stack(image_sizes, dim=0),
            }
            return inputs_batch, torch.tensor(label_list,dtype=torch.long)

class GeneralRewardDataset_llava(Dataset):
    """
    General dataset for reward model, handling both custom and standard formats.

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        is_custom: flag indicating whether the dataset is in custom format
    """
    def __init__(
        self,
        dataset,
        processor: Callable,
        tokenizer: Callable,
        # max_length: int,
        strategy,
        is_custom=False,
        return_prompt_length=False,
        cls_based=False,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.tokenizer = tokenizer
        self.processor = processor
        self.strategy = strategy
        self.is_custom = is_custom
        self.return_prompt_length = return_prompt_length
        self.cls_based=cls_based
        if self.cls_based:
            self.path_list = []
            self.label_list = []
            for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
                prompt, path, label = preprocess_data_cls(data)
                self.prompts.append(prompt)
                self.path_list.append(path)
                self.label_list.append(label)
        else:
            self.chosens = []
            self.rejects = []
            self.c_rates = []
            self.r_rates = []
            for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
                prompt, chosen, reject,c_rate,r_rate = preprocess_data(data)
                self.prompts.append(prompt)
                self.chosens.append(chosen)
                self.rejects.append(reject)
                self.c_rates.append(c_rate)
                self.r_rates.append(r_rate)
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if not self.cls_based:
            prompt, chosen, reject,c_rate,r_rate= self.prompts[idx], self.chosens[idx], self.rejects[idx],self.c_rates[idx],self.r_rates[idx]
            chosen_img= Image.open(chosen).convert("RGB")
            reject_img= Image.open(reject).convert("RGB")
            if not isinstance(prompt, list):
                prompt_c=prompt
                prompt_r=prompt
            else:
                prompt_c=prompt[0]
                prompt_r=prompt[1]
            prompt_message_c =[
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_c},
                    {"type": "image"},
                    ],
                },
            ]
            text_c = self.processor.apply_chat_template(prompt_message_c, tokenize=False, add_generation_prompt=True)[0:-11]+self.tokenizer.eos_token
            prompt_message_r = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_r},
                    {"type": "image"},
                    ],
                },
            ]
            text_r = self.processor.apply_chat_template(prompt_message_r, tokenize=False, add_generation_prompt=True)[0:-11]+self.tokenizer.eos_token      
            return (
                chosen_img,
                text_c,
                reject_img,
                text_r,
                c_rate,
                r_rate
            )
        else:
            prompt = self.prompts[idx]
            path, label = self.path_list[idx], self.label_list[idx]
            img = Image.open(path).convert("RGB")

            prompt_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }
            ]
            text = (
                self.processor.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)[
                    0:-11
                ]
                + self.tokenizer.eos_token
            )
            return img, text, label

    def collate_fn(self, item_list):
        if not self.cls_based:
            c_rate_list=[]
            r_rate_list=[]
            img_batch_c=[]
            img_batch_r=[]
            text_batch_c=[]
            text_batch_r=[]
            for chosen_img, text_c,reject_img, text_r,c_rate,r_rate in item_list:
                img_batch_c.append(chosen_img)
                img_batch_r.append(reject_img)
                text_batch_c.append(text_c)
                text_batch_r.append(text_r)
                c_rate_list.append(c_rate)
                r_rate_list.append(r_rate)
            inputs_batch_c = self.processor(
                images=img_batch_c,
                text=text_batch_c,
                padding=True,
                return_tensors="pt",
            )
            inputs_batch_r = self.processor(
                images=img_batch_r,
                text=text_batch_r,
                padding=True,
                return_tensors="pt",
            )
            return inputs_batch_c, inputs_batch_r,c_rate_list,r_rate_list
        else:
            imgs, txts, labels = zip(*item_list)
            inputs_batch = self.processor(images=list(imgs), text=list(txts), padding=True, return_tensors="pt")
            return inputs_batch, torch.tensor(labels, dtype=torch.long)

class GeneralRewardDataset_qwen(Dataset):
    """
    General dataset for reward model, handling both custom and standard formats.

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        is_custom: flag indicating whether the dataset is in custom format
    """
    def __init__(
        self,
        dataset,
        processor: Callable,
        tokenizer: Callable,
        # max_length: int,
        strategy,
        is_custom=False,
        return_prompt_length=False,
        cls_based=False,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.tokenizer = tokenizer
        self.processor = processor
        self.strategy = strategy
        self.is_custom = is_custom
        self.return_prompt_length = return_prompt_length
        self.cls_based=cls_based
        if self.cls_based:
            self.path_list = []
            self.label_list = []
            for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
                prompt, path, label = preprocess_data_cls(data)
                self.prompts.append(prompt)
                self.path_list.append(path)
                self.label_list.append(label)
        else:
            self.chosens = []
            self.rejects = []
            self.c_rates = []
            self.r_rates = []
            for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
                prompt, chosen, reject,c_rate,r_rate = preprocess_data(data)
                self.prompts.append(prompt)
                self.chosens.append(chosen)
                self.rejects.append(reject)
                self.c_rates.append(c_rate)
                self.r_rates.append(r_rate)
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if not self.cls_based:
            prompt, chosen, reject,c_rate,r_rate= self.prompts[idx], self.chosens[idx], self.rejects[idx],self.c_rates[idx],self.r_rates[idx]
            if not isinstance(prompt, list):
                prompt_c=prompt
                prompt_r=prompt
            else:
                prompt_c=prompt[0]
                prompt_r=prompt[1]
            prompt_message_c = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "file://"+chosen,
                        },
                        {"type": "text", "text": prompt_c},
                    ],
                }
            ]
            text_c = self.processor.apply_chat_template(prompt_message_c, tokenize=False, add_generation_prompt=True)[58:-23].strip()
            prompt_message_r = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "file://"+reject,
                        },
                        {"type": "text", "text": prompt_r},
                    ],
                }
            ]
            text_r = self.processor.apply_chat_template(prompt_message_r, tokenize=False, add_generation_prompt=True)[58:-23].strip()
            return (
                prompt_message_c,
                text_c,
                prompt_message_r,
                text_r,
                c_rate,
                r_rate
            )
        else:
            prompt = self.prompts[idx]
            path, label = self.path_list[idx], self.label_list[idx]

            prompt_msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file://" + path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)[58:-23].strip()
            return prompt_msg, text, label

    def collate_fn(self, item_list):
        if not self.cls_based:
            c_rate_list=[]
            r_rate_list=[]
            messages_batch_c=[]
            messages_batch_r=[]
            text_batch_c=[]
            text_batch_r=[]
            for prompt_message_c, text_c,prompt_message_r, text_r,c_rate,r_rate in item_list:
                messages_batch_c.append(prompt_message_c)
                messages_batch_r.append(prompt_message_r)
                text_batch_c.append(text_c)
                text_batch_r.append(text_r)
                c_rate_list.append(c_rate)
                r_rate_list.append(r_rate)
            image_inputs_c, video_inputs_c = process_vision_info(messages_batch_c)
            image_inputs_r, video_inputs_r = process_vision_info(messages_batch_r)
            inputs_batch_c = self.processor(
                text=text_batch_c,
                images=image_inputs_c,
                videos=video_inputs_c,
                padding=True,
                return_tensors="pt",
            )
            inputs_batch_r = self.processor(
                text=text_batch_r,
                images=image_inputs_r,
                videos=video_inputs_r,
                padding=True,
                return_tensors="pt",
            )
            return inputs_batch_c, inputs_batch_r,c_rate_list,r_rate_list
        else:
            msgs, txts, labels = zip(*item_list)
            imgs, vids = process_vision_info(list(msgs))
            inputs_batch = self.processor(text=list(txts), images=imgs, videos=vids, padding=True, return_tensors="pt")
            return inputs_batch, torch.tensor(labels, dtype=torch.long)
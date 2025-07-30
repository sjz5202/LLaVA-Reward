# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/rm_trainer.py

from abc import ABC
import os 
import shutil
import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from llava_reward.utils.custom_distributed_sampler import GroupDistributedSampler
from tqdm import tqdm
import deepspeed
from llava_reward.models import PairWiseLoss,FocalPairWiseLoss, GeneralPreferenceLoss, GeneralPreferenceLoss_no_R, HighDimGeneralPreferenceLoss, SFTMeanLoss, SFTSumLoss, DPORefFreeLoss, SFTVanillaLoss
from llava_reward.models import GeneralPreferenceLearnableTauLoss, GeneralPreferenceLearnableTauRegressionLoss, GeneralPreferenceRegressionLoss
from llava_reward.models import PairWiseLearnableTauLoss, PairWiseLearnableTauRegressionLoss, PairWiseRegressionLoss, HighDimGeneralPreferenceRegressionMoELoss
from llava_reward.models import HighDimGeneralPreferenceRegressionLoss, HighDimGeneralPreferenceMoELoss,Cls_loss,Binary_Cls_loss
class GeneralPreferenceRewardTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        is_general_preference (bool, defaults to False): Whether the model is a General Preference model.
        tau (float, defaults to 0.1): Hyperparameter tau used in the calculation of General Preference loss.
        value_head_dim (int, defaults to 2): Dimension of the value head in the General Preference model. Ignored by the Bradley Terry model.

    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_epochs: int = 2,
        is_general_preference: bool = False,
        add_unpaired_head: bool = False,
        add_cross_attention: bool = False,
        cls_based: bool = False,
        add_img_eos: bool = False,
        tau: float = 0.1,
        value_head_dim: int = 2,
        
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.is_general_preference = is_general_preference
        self.cls_based = cls_based
        self.add_cross_attention=add_cross_attention
        self.tau=tau
        self.value_head_dim=value_head_dim
        if is_general_preference:
            if value_head_dim == 2 and not self.args.add_prompt_head and not add_img_eos:
                self.loss_fn =GeneralPreferenceLoss(tau)
                self.strategy.print("GeneralPreference Loss")
            elif value_head_dim == 1:
                self.loss_fn = Cls_loss()
                self.strategy.print("Cls_loss Loss")
            else:
                assert value_head_dim % 2 == 0, "Dimension of value head for general preference model can not be odd!"
                if self.args.add_prompt_head:
                    self.loss_fn = HighDimGeneralPreferenceMoELoss(model=self.model, value_head_dim=value_head_dim, softmax_tau=tau)
                else:
                    self.loss_fn = HighDimGeneralPreferenceLoss(tau, value_head_dim)
                    if add_img_eos:
                        self.loss_fn =GeneralPreferenceLoss(tau)
                        self.strategy.print("add_img_eos GeneralPreference Loss")
        elif cls_based:
            # unpaired CLS loss
            self.loss_fn = Binary_Cls_loss()
            self.strategy.print("Binary cls Loss")
        else:
            # BT
            self.loss_fn = PairWiseLoss(tau=tau)
            self.strategy.print("PairWiseLoss Loss")

        self.ptx_loss_fn = SFTSumLoss(self.args.reward_scaler_beta)
        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb
            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        reward_config={"general_preference_tau":self.tau,"value_head_dim":self.value_head_dim,'add_cross_attention':self.add_cross_attention,'is_general_preference':self.is_general_preference}
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        eval_loss_minimum = None
        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )
                
            if isinstance(self.train_dataloader.sampler, GroupDistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            loss_mean = 0
            for inputs_batch_c, inputs_batch_r,c_rate_list,r_rate_list in self.train_dataloader:
                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                if self.model.model_type == 'phi3v':
                    chosen_ids = inputs_batch_c['input_ids'].squeeze(1).to(torch.cuda.current_device())
                    c_mask = inputs_batch_c['attention_mask'].squeeze(1).to(torch.cuda.current_device())
                    c_pixel_value=inputs_batch_c['pixel_values'].squeeze(1).to(torch.cuda.current_device())
                    c_img_size=inputs_batch_c['image_sizes'].squeeze(1).to(torch.cuda.current_device())

                    reject_ids = inputs_batch_r['input_ids'].squeeze(1).to(torch.cuda.current_device())
                    r_mask = inputs_batch_r['attention_mask'].squeeze(1).to(torch.cuda.current_device())
                    r_pixel_value=inputs_batch_r['pixel_values'].squeeze(1).to(torch.cuda.current_device())
                    r_img_size=inputs_batch_r['image_sizes'].squeeze(1).to(torch.cuda.current_device())
                    return_output = True if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss) or isinstance(self.loss_fn, HighDimGeneralPreferenceMoELoss) else False
                    chosen_reward, reject_reward, outputs = self.concatenated_forward(self.model, chosen_ids, c_mask,c_pixel_value, c_img_size, reject_ids, r_mask, r_pixel_value,r_img_size,return_output=return_output)
                else:
                    inputs_batch_c=inputs_batch_c.to(torch.cuda.current_device())
                    inputs_batch_r=inputs_batch_r.to(torch.cuda.current_device())
                    return_output = True if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss) or isinstance(self.loss_fn, HighDimGeneralPreferenceMoELoss) else False
                    chosen_reward, reject_reward, outputs = self.concatenated_forward(model=self.model, inputs_batch_c=inputs_batch_c, inputs_batch_r=inputs_batch_r,return_output=return_output)
                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()

                if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss) or isinstance(self.loss_fn, HighDimGeneralPreferenceMoELoss):
                    chosen_last_hidden_states = outputs["last_hidden_state"][: chosen_ids.shape[0], :, :]
                    prompt_end_index = chosen_last_hidden_states.size(1) - chosen_response_len - 1
                    prompt_end_index_expanded = prompt_end_index.unsqueeze(-1).expand(-1, -1, chosen_last_hidden_states.size(-1))
                    prompt_hidden_state = torch.gather(chosen_last_hidden_states, dim=1, index=prompt_end_index_expanded).squeeze(1)
                    preference_loss, prob = self.loss_fn(chosen_reward, reject_reward, prompt_hidden_state.to(torch.cuda.current_device()), margin)
                else:
                    preference_loss, prob = self.loss_fn(chosen_reward, reject_reward, margin)
                
                if args.add_pretrain_loss:
                    if isinstance(self.ptx_loss_fn, DPORefFreeLoss):
                        chosen_output = self.model.forward(chosen_ids, attention_mask=c_mask)
                        chosen_label = torch.where(
                            c_mask.bool(),
                            chosen_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        chosen_log_probs = chosen_output["logits"]
                        rejected_output = self.model.forward(reject_ids, attention_mask=r_mask)
                        rejected_label = torch.where(
                            r_mask.bool(),
                            reject_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        rejected_log_probs = rejected_output["logits"] 
                        chosen_reward_ptx_loss = self.ptx_loss_fn(chosen_log_probs, chosen_label, c_mask.bool(), rejected_log_probs, rejected_label, r_mask.bool())
                    else:
                        ptx_output = self.model.forward(chosen_ids, attention_mask=c_mask,pixel_values=c_pixel_value ,image_sizes=c_img_size)
                        c_mask = c_mask & (chosen_ids != -1)
                        ptx_label = torch.where(
                            c_mask.bool(),
                            chosen_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        ptx_log_probs = ptx_output["logits"]
                        chosen_reward_ptx_loss = self.ptx_loss_fn(ptx_log_probs, ptx_label, c_mask.bool())
                        chosen_reward_ptx_loss=preference_loss
                    loss = (1 - args.ptx_loss_coef) * preference_loss + chosen_reward_ptx_loss * args.ptx_loss_coef
                else:
                    loss = preference_loss

                del chosen_reward, reject_reward, outputs
                torch.cuda.empty_cache()
                
                self.strategy.backward(loss, self.model, self.optimizer)
                
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                
                logs_dict = {
                    "preference_loss": preference_loss.item(),
                    "prob": prob.item(),
                    "loss_mean": loss_mean,
                }
                    
                # logs/checkpoints/evaluate
                eval_loss_minimum = self.save_logs_and_checkpoints(args, global_step, step_bar, epoch, logs_dict,reward_config=reward_config)
                torch.distributed.barrier()
                step_bar.update()
                global_step += 1
            tag = f"epoch_{epoch}"
            if args.lora_rank>0:
                self.strategy.save_model_lora(self.model, self.tokenizer, self.add_cross_attention,os.path.join(args.save_path, tag),reward_config=reward_config)  
            else:
                self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.save_path, tag))
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
            
    def cls_fit(self, args):
        # get eval and save steps
        reward_config={"general_preference_tau":self.tau,"value_head_dim":self.value_head_dim,'add_cross_attention':self.add_cross_attention,'is_general_preference':self.is_general_preference}
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        eval_loss_minimum = None
        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )
                
            if isinstance(self.train_dataloader.sampler, GroupDistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            loss_mean = 0
            for inputs_batch, labels in self.train_dataloader:
                labels=labels.to(torch.cuda.current_device())
                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                if self.model.model_type == 'phi3v':
                    ids = inputs_batch['input_ids'].squeeze(1).to(torch.cuda.current_device())
                    masks = inputs_batch['attention_mask'].squeeze(1).to(torch.cuda.current_device())
                    pixel_values=inputs_batch['pixel_values'].squeeze(1).to(torch.cuda.current_device())
                    img_sizes=inputs_batch['image_sizes'].squeeze(1).to(torch.cuda.current_device())
                    labels=labels.to(torch.cuda.current_device())
                    return_output = True if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss) or isinstance(self.loss_fn, HighDimGeneralPreferenceMoELoss) else False
                    rewards, outputs = self.cls_forward(self.model, ids=ids, masks=masks,pixel_values=pixel_values, img_sizes=img_sizes, labels=labels,return_output=return_output)
                else:
                    inputs_batch=inputs_batch.to(torch.cuda.current_device())
                    return_output = True if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss) or isinstance(self.loss_fn, HighDimGeneralPreferenceMoELoss) else False
                    rewards, outputs  = self.cls_forward(self.model, inputs_batch=inputs_batch,return_output=return_output)
                # loss function
                if self.compute_fp32_loss:
                    rewards = rewards.float()
                preference_loss, prob = self.loss_fn(rewards,labels,margin)
                
                if args.add_pretrain_loss:
                    if isinstance(self.ptx_loss_fn, DPORefFreeLoss):
                        chosen_output = self.model.forward(chosen_ids, attention_mask=c_mask)
                        chosen_label = torch.where(
                            c_mask.bool(),
                            chosen_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        chosen_log_probs = chosen_output["logits"]
                        rejected_output = self.model.forward(reject_ids, attention_mask=r_mask)
                        rejected_label = torch.where(
                            r_mask.bool(),
                            reject_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        rejected_log_probs = rejected_output["logits"] 
                        chosen_reward_ptx_loss = self.ptx_loss_fn(chosen_log_probs, chosen_label, c_mask.bool(), rejected_log_probs, rejected_label, r_mask.bool())
                    else:
                        ptx_output = self.model.forward(chosen_ids, attention_mask=c_mask,pixel_values=c_pixel_value ,image_sizes=c_img_size)
                        c_mask = c_mask & (chosen_ids != -1)
                        ptx_label = torch.where(
                            c_mask.bool(),
                            chosen_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        ptx_log_probs = ptx_output["logits"]
                        chosen_reward_ptx_loss = self.ptx_loss_fn(ptx_log_probs, ptx_label, c_mask.bool())
                        chosen_reward_ptx_loss=preference_loss
                    loss = (1 - args.ptx_loss_coef) * preference_loss + chosen_reward_ptx_loss * args.ptx_loss_coef
                else:
                    loss = preference_loss

                
                self.strategy.backward(loss, self.model, self.optimizer)
                
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                
                logs_dict = {
                    "preference_loss": preference_loss.item(),
                    "prob": prob.item(),
                    "loss_mean": loss_mean,
                }
                    
                # logs/checkpoints/evaluate
                eval_loss_minimum = self.save_logs_and_checkpoints(args, global_step, step_bar, epoch, logs_dict,reward_config=reward_config)
                torch.distributed.barrier()
                step_bar.update()
                global_step += 1
            tag = f"epoch_{epoch}"
            if args.lora_rank>0:
                self.strategy.save_model_lora(self.model, self.tokenizer,self.add_cross_attention, os.path.join(args.save_path, tag),reward_config=reward_config)  
            else:
                self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.save_path, tag))
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
    def save_logs_and_checkpoints(self, args, global_step, step_bar, epoch, logs_dict={},model_type='phi',reward_config={}):
        if global_step % args.logging_steps == 0:
            # Reduce and log training loss and other metrics
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # Log to Weights & Biases if enabled
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # Save the model at specified save steps without evaluation
        if global_step % args.save_steps == 0:
            tag = f"epoch_{epoch}_global_step_{global_step}"
            if args.lora_rank>0:
                self.strategy.save_model_lora(self.model, self.tokenizer, self.add_cross_attention,os.path.join(args.save_path, tag),reward_config)
            else:
                self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.save_path, tag))
            # Log saving action if needed
            self.strategy.print(f"Model saved at step {global_step}")

        # Always return None as eval_loss_minimum is no longer relevant
        return None
    
    def clean_old_checkpoints(self, output_dir, max_checkpoints=3): 
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('step_')] # If the number of directories exceeds max_checkpoints, delete the oldest one 
        if len(subdirs) > max_checkpoints: 
            subdirs.sort(key=lambda x: int(x.split('_')[1])) 
            dir_to_delete = os.path.join(output_dir, subdirs[0]) 
            try:
                shutil.rmtree(dir_to_delete) 
            except Exception as e:
                print(f"Error deleting old checkpoint{dir_to_delete}: {e}")


    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0
            prob_sum = 0
            for chosen_ids, c_mask,c_pixel_value,c_img_size, reject_ids, r_mask, r_pixel_value,r_img_size in eval_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                c_pixel_value=c_pixel_value.squeeze(1).to(torch.cuda.current_device())
                c_img_size=c_img_size.squeeze(1).to(torch.cuda.current_device())

                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                r_pixel_value=r_pixel_value.squeeze(1).to(torch.cuda.current_device())
                r_img_size=r_img_size.squeeze(1).to(torch.cuda.current_device())

                return_output = True if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss) else False
                chosen_reward, reject_reward, outputs = self.concatenated_forward(
                    self.model, chosen_ids, c_mask,c_pixel_value, c_img_size, reject_ids, r_mask, r_pixel_value,r_img_size,return_output
                )
                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss):
                    chosen_last_hidden_states = outputs["last_hidden_state"][: chosen_ids.shape[0], :, :]
                    prompt_len = chosen_last_hidden_states.size(1) - chosen_response_len
                    prompt_len_expanded = prompt_len.unsqueeze(-1).expand(-1, -1, chosen_last_hidden_states.size(-1))
                    prompt_hidden_state = torch.gather(chosen_last_hidden_states, dim=1, index=prompt_len_expanded).squeeze(1)
                    preference_loss, prob = self.loss_fn(chosen_reward, reject_reward, prompt_hidden_state, margin)
                else:
                    preference_loss, prob = self.loss_fn(chosen_reward, reject_reward, margin)
                    
                loss = preference_loss

                loss_sum += loss.item() 
                prob_sum += prob.item() 
                  
                step_bar.update()

            loss_mean = loss_sum / self.eval_dataloader.__len__()
            prob_mean = prob_sum / self.eval_dataloader.__len__()

            bar_dict = {
                "eval_loss_mean": loss_mean,
                "prob_mean": prob_mean,
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)

        self.model.train()  # reset model state
        torch.cuda.empty_cache() 
        if self.strategy.is_rank_0():  
            return loss_mean

    def concatenated_forward(self, model, chosen_ids=None, c_mask=None,c_pixel_value=None,c_img_size=None, reject_ids=None, r_mask=None,r_pixel_value=None,r_img_size=None, 
        inputs_batch_c=None, inputs_batch_r=None,
        return_output: bool = False):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        """
        model = model.module if hasattr(model, "module") else model
        if model.model_type == 'phi3v':
            chosen_rewards, c_outputs = model.custom_forward(input_ids=chosen_ids, attention_mask=c_mask,pixel_values=c_pixel_value ,image_sizes=c_img_size,return_output=return_output)
            rejected_rewards, r_outputs = model.custom_forward(input_ids=reject_ids, attention_mask=r_mask,pixel_values=r_pixel_value ,image_sizes=r_img_size,return_output=return_output)
        else:
            chosen_rewards, c_outputs = model.custom_forward(inputs_batch=inputs_batch_c,return_output=return_output)
            rejected_rewards, r_outputs = model.custom_forward(inputs_batch=inputs_batch_r,return_output=return_output) 
        #<end>  <end_of_text> 3071 -> 2  ->Ｂ＊ＳＥＱ * 2 
        return chosen_rewards, rejected_rewards, [c_outputs,r_outputs]
        
    def concatenated_forward_qwen(self,model, inputs_batch_c, inputs_batch_r,return_output: bool = False):
        model = model.module if hasattr(model, "module") else model
        chosen_rewards, c_outputs = model.custom_forward(inputs_batch=inputs_batch_c,return_output=return_output,model_type='qwen')
        rejected_rewards, r_outputs = model.custom_forward(inputs_batch=inputs_batch_r,return_output=return_output,model_type='qwen')        
        return chosen_rewards, rejected_rewards, [c_outputs,r_outputs]
    def concatenated_forward_llava(self,model, inputs_batch_c, inputs_batch_r,return_output: bool = False):
        model = model.module if hasattr(model, "module") else model
        chosen_rewards, c_outputs = model.custom_forward(inputs_batch=inputs_batch_c,return_output=return_output,model_type='llava')
        rejected_rewards, r_outputs = model.custom_forward(inputs_batch=inputs_batch_r,return_output=return_output,model_type='llava')        
        return chosen_rewards, rejected_rewards, [c_outputs,r_outputs]

    def cls_forward(self, model, ids=None, masks=None,pixel_values=None,img_sizes=None, inputs_batch=None,labels=None, return_output: bool = False):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        """
        model = model.module if hasattr(model, "module") else model
        if model.model_type == 'phi3v':
            rewards, outputs = model.custom_forward(input_ids=ids, attention_mask=masks,pixel_values=pixel_values ,image_sizes=img_sizes,return_output=return_output)
        else:
            rewards, outputs = model.custom_forward(inputs_batch=inputs_batch,return_output=return_output)
        #<end>  <end_of_text> 3071 -> 2  ->Ｂ＊ＳＥＱ * 2 
        return rewards,outputs
    

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks

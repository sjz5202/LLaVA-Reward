import torch
from eval.reward_adaptor_loader import load_reward_adaptor, inference_process_phi3v, preference_compute
import os

class Args:
    pass
args = Args()
args.pm_path = "/code/llava-reward-ckpt/alignment/llavareward_phi_alignment"
args.pretrain = "microsoft/Phi-3.5-vision-instruct"
args.cache_dir = None
args.ft_projector = True
args.seed = 1234
args.disable_fast_tokenizer = False

# load model
args, model, processor, tokenizer = load_reward_adaptor(args,model_type='phi3v',reward_config_path=os.path.join(args.pm_path, "reward_config.yaml"),load_tokenizer=True)
model.to('cuda')
model.eval()

# prepare example
caption = "perfect white haired egyptian goddess wearing white dove wings, warframe armor, regal, attractive, ornate, sultry, beautiful, ice queen, half asian, pretty face, blue eyes, detailed, scifi platform, 4 k, ultra realistic, epic lighting, illuminated, cinematic, masterpiece, art by akihito tsukushi, voidstar"
img_dir_list = ["data/sample_test/sample_img/0_1_id_000904-0035.jpg", "data/sample_test/sample_img/4_3_id_000904-0035.jpg"]
img_inputs = inference_process_phi3v(args,processor,tokenizer,img_dir_list,caption,device='cuda')
img_inputs_c = img_inputs[0]
img_inputs_r = img_inputs[1]

# inference
with torch.no_grad():
    chosen_rewards, _ = model.custom_forward(**img_inputs_c)
    reject_rewards, _ = model.custom_forward(**img_inputs_r)
prob = preference_compute(args,chosen_rewards,reject_rewards)

if not args.is_general_preference:
    print("image0 reward:", chosen_rewards.item())
    print("image1 reward:", reject_rewards.item())
print('Predict probability that image0 is better than image1:',prob)
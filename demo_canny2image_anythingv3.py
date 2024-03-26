from share import *
import config
from cldm.hack import hack_everything

hack_everything(clip_skip=2)

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_any3_canny.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

img_dir = '/nvme/liuwenran/datasets/others/caixukun.png'
img = cv2.imread(img_dir)
prompt = 'a girl, smoking, sunny, summer, good light'
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
# a_prompt = ''
# n_prompt = ''
num_samples = 1
image_resolution = 512
low_threshold = 100
high_threhold = 200
ddim_steps = 20
scale = 9.0
seed = 1
guess_mode = False
strength = 0.1
eta = 0.0

save_folder = 'results/anythingv3_canny_zhou_zenmela_sunny'
import os
if not os.path.exists(save_folder):
   os.makedirs(save_folder)

frame_file = '/nvme/liuwenran/datasets/zhou_zenmela_frames/frames.txt'
lines = open(frame_file, 'r').readlines()

for ind in range(len(lines)):
    img_dir = lines[ind].strip()
    img = cv2.imread(img_dir)
    results = process(img, prompt=prompt, a_prompt=a_prompt, n_prompt=n_prompt, num_samples=num_samples, image_resolution=image_resolution,
                  low_threshold=low_threshold, high_threshold=high_threhold, ddim_steps=ddim_steps, guess_mode=guess_mode, strength=1.0, scale=scale,
                  seed=seed, eta=eta)
    cv2.imwrite(os.path.join(save_folder, 'res_{:0>4d}.jpg'.format(ind)), results[1])

video_name = save_folder.split('/')[-1]
cmd = 'ffmpeg -r 30 -i ' + save_folder + '/res_%04d.jpg -b:v 30M -vf fps=30 results/' + video_name + '.mp4'
os.system(cmd)

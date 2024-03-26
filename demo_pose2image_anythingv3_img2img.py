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
import PIL
from PIL import Image

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

apply_openpose = OpenposeDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_any3_openpose.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, init_latent, save_control_frame_ind, save_folder):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # guess_mode = True

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # strength = 0.0
        # model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        # model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  

        model.control_scales = [0.0] * 12 + [1.0]
        # import ipdb;ipdb.set_trace();

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond,
                                                     x_T=init_latent,
                                                     save_control_frame_ind=save_control_frame_ind,
                                                     save_folder=save_folder)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # save detected map
        detected_map_save_folder = os.path.join(save_folder, 'detected_map')
        if not os.path.exists(detected_map_save_folder):
            os.makedirs(detected_map_save_folder)
        
        cv2.imwrite(os.path.join(detected_map_save_folder, '{:0>4d}.jpg'.format(ind)), detected_map)
        # save detected map


        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

prompt = 'a handsome man, silver hair, gray coat, smiling face, a goodlooking necklace, no background' 
#             'a handsome man, silver hair, gray coat, smiling face, a goodlooking necklace, no background' + \
#             'a handsome man, silver hair, gray coat, smiling face, a goodlooking necklace, no background' + \
#             'a handsome man, silver hair, gray coat, smiling face, a goodlooking necklace, no background'
# prompt = 'a girl, black hair, smoking'
a_prompt = 'best quality, extremely detailed, no background'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
num_samples = 1
image_resolution = 512
detect_resolution = 512
ddim_steps = 20
scale = 9.0
seed = 202303
guess_mode = False
strength = 1.0
eta = 0.0


img_dir = '/nvme/liuwenran/datasets/others/caixukun.png'
img = cv2.imread(img_dir)
# save_folder = 'results/anythingv3_zhou_zenmela_pose_debug'
# save_folder = 'results/anythingv3_zhou_zenmela_fps10_pose_debug_onlymid1_mom0_tsle3'
save_folder = 'results/anythingv3_pose_caixunkun_dancing_begin_fps30_debug_onlymid1_mom0_ts5'
# save_folder = 'results/anythingv3_pose_caixukun_dance_begin_fixallnoise_alwaysfirst'
import os
if not os.path.exists(save_folder):
   os.makedirs(save_folder)

# frame_file = '/nvme/liuwenran/datasets/zhou_zenmela_fps10_frames/frames.txt'
frame_file = '/nvme/liuwenran/datasets/caixukun_dancing_begin_fps30_frames/frames.txt'
lines = open(frame_file, 'r').readlines()

seed_everything(seed)
init_latent = None


use_img2img = False
if use_img2img:
    latent_strength = 0.9
    latent_img_dir  = '/nvme/liuwenran/repos/ControlNet/results/anythingv3_pose_caixukun_dance_begin_manyprompt/res_0.jpg'
    latent_img = cv2.imread(latent_img_dir)
    latent_img = load_img(latent_img_dir).cuda()
    # init_latent = model.get_first_stage_encoding(model.encode_first_stage(latent_img))  # move to latent space
    ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
    t_enc = int(latent_strength * ddim_steps)
    t_enc = torch.tensor([t_enc]).cuda()
    # init_latent = ddim_sampler.stochastic_encode(init_latent, t_enc)


use_init_noise = True
if use_init_noise:
    img_dir = lines[0].strip()
    img = cv2.imread(img_dir)
    img = resize_image(img, image_resolution)
    H, W, C = img.shape
    init_noise_shape = (len(lines), 4,  H // 8, W // 8)
    init_noise_all_frame = torch.randn(init_noise_shape).cuda()


for ind in range(len(lines)):
    if use_init_noise:
        init_latent = init_noise_all_frame[0].unsqueeze(0)

    img_dir = lines[ind].strip()
    img = cv2.imread(img_dir)
    results = process(img, prompt=prompt, a_prompt=a_prompt, n_prompt=n_prompt, num_samples=num_samples, image_resolution=image_resolution,
                    detect_resolution=detect_resolution, ddim_steps=ddim_steps, guess_mode=guess_mode, strength=1.0, scale=scale,
                    seed=seed, eta=eta, init_latent=init_latent, save_control_frame_ind=ind, save_folder=save_folder)
    cv2.imwrite(os.path.join(save_folder, 'res_{:0>4d}.jpg'.format(ind)), results[1])

    if use_img2img and init_latent is None:
        latent_img = results[1].astype(np.float32) / 255.0
        latent_img = latent_img[None].transpose(0, 3, 1, 2)
        latent_img = torch.from_numpy(latent_img)
        latent_img = 2. * latent_img - 1
        latent_img = latent_img.cuda()

        init_latent = model.get_first_stage_encoding(model.encode_first_stage(latent_img))  # move to latent space
        init_latent = ddim_sampler.stochastic_encode(init_latent, t_enc)

video_name = save_folder.split('/')[-1]
cmd = 'ffmpeg -r 60 -i ' + save_folder + '/res_%04d.jpg -b:v 30M -vf fps=10 results/' + video_name + '.mp4'
os.system(cmd)

import os
import random
import sys
from argparse import ArgumentParser
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from k_diffusion import utils
from k_diffusion.sampling import default_noise_sampler, to_d, get_ancestral_step
from tqdm.auto import trange
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import clip

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json

def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    extra_args = extra_args or {}
    noise_sampler = noise_sampler or default_noise_sampler(x)
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

def resize_image_to_resolution(input_image, resolution, reverse=True):
    width, height = input_image.size
    scale = resolution / min(width, height) if reverse else resolution / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    return ImageOps.fit(input_image, (new_width, new_height), method=Image.Resampling.LANCZOS)

class CFGDenoiser(nn.Module):
    def __init__(self, model, contrastive_loss):
        super().__init__()
        self.inner_model = model
        self.clip_model, self.clip_preprocess = clip.load('ViT-L/14', jit=False)
        self.clip_model = self.clip_model.eval().requires_grad_(False).to('cuda:1')
        self.contrastive_loss = contrastive_loss

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, input_image=None):
        
        if self.contrastive_loss:
            z = self.clip_loss(z, sigma, cond, input_image)
            z = z.detach()
            
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_txt_cond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return 0.5 * (out_img_cond + out_txt_cond) + text_cfg_scale * (out_cond - out_img_cond) +\
              image_cfg_scale * (out_cond - out_txt_cond)

    def global_loss(self, input_img, target_prompt):
        text_target_tokens = clip.tokenize(target_prompt).to('cuda:1')
        similarity = 1 - self.clip_model(input_img, text_target_tokens)[0] / 100
        return similarity.mean()

    def directional_loss(self, input_img, timesteps_img, source_prompt, target_prompt):
        
        encoded_image_diff = input_img - timesteps_img
        encoded_text_diff = source_prompt - target_prompt
        cosine_similarity = F.cosine_similarity(encoded_image_diff, encoded_text_diff, dim= -1)

        return (1 - cosine_similarity).mean()
    
    def clip_loss(self, z, sigma, cond, input_img):
        t = self.inner_model.sigma_to_t(sigma).long()
        alpha = 1

        if t < 400:
            return z
    
        with torch.set_grad_enabled(True):
            z = z.requires_grad_(True)

            # 1. Transform latent (z) to image
            x_pred = self.inner_model.inner_model.differentiable_decode_first_stage(z).requires_grad_(True)
            # save image
            pixel_img = torch.clamp((x_pred + 1.0) / 2.0, min=0.0, max=1.0)
            pixel_img = 255.0 * rearrange(pixel_img, "1 c h w -> h w c")
            original_image = Image.fromarray(pixel_img.type(torch.uint8).cpu().numpy())
            original_image.save(os.path.join("./timestep", f"{t}.jpg"))
            
            # 2. Encode image with CLIP image encoder
            resize = transforms.Resize((224, 224))
            x_pred = resize(x_pred).to("cuda:1")
            image_embedding = self.clip_model.encode_image(x_pred).float().requires_grad_(True)
            input_img = resize(input_img).to("cuda:1")
            input_img_embedding = self.clip_model.encode_image(input_img).float().requires_grad_(True)

            target_text_embedding = cond["c_crossattn"][0][0].to("cuda:1")
            source_text_embedding = cond["c_crossattn"][0][1].to("cuda:1")
            
            g_loss = self.global_loss(input_img, cond["inst_prompt"])
            dir_loss = self.directional_loss(
                input_img_embedding, 
                image_embedding, 
                source_text_embedding, 
                target_text_embedding)

            loss = g_loss * 300 + dir_loss * 300
            gradient = torch.autograd.grad(loss.to("cuda:0"), [z])[0]
       
        z = z - alpha * gradient * 100
        z = z.detach()
        torch.cuda.empty_cache()
        del gradient, loss, target_text_embedding, source_text_embedding, image_embedding, input_img_embedding
        return z
    
        
def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd:
        pl_sd = pl_sd['state_dict']
    m, u = model.load_state_dict(pl_sd, strict=False)
    print(m, u)
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/promptfix.yaml", type=str)
    parser.add_argument("--ckpt", default="./checkpoints/promptfix.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--indir", default='./examples/validation', type=str)
    parser.add_argument("--outdir", default="validation_results/", type=str)
    parser.add_argument("--cfg-text", default=6.5, type=float)
    parser.add_argument("--cfg-image", default=1.25, type=float)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--disable_hf_guidance", type=bool, default=True)
    parser.add_argument("--enable-flaw-prompt", type=bool, default=True)
    parser.add_argument("--contrastive-loss", default= True, type=bool)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    os.makedirs(args.outdir, exist_ok=True)

    model = load_model_from_config(config, args.ckpt, args.vae_ckpt).to("cuda:0")
    model.eval().to('cuda:0')

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap, args.contrastive_loss)
    null_token = model.get_learned_conditioning([""])

    seed = args.seed if args.seed is not None else random.randint(0, 100000)
    
    instruct_dic = json.load(open(os.path.join(args.indir, 'instructions.json')))
    
    for val_img_idx, image_path in enumerate(instruct_dic):
        print(image_path, instruct_dic[image_path])
        
        input_image = Image.open(os.path.join(args.indir, image_path)).convert("RGB")
        input_image = resize_image_to_resolution(input_image, args.resolution, 'inpaint' not in image_path)
        input_image_pil = input_image

        with autocast("cuda:0"):
            cond = {}
            inst_prompt, flaw_prompt = instruct_dic[image_path]
            
            cond["inst_prompt"] = inst_prompt
            cond["flaw_prompt"] = flaw_prompt
            
            # instruction prompt and flaw prompt embedding 
            cond["c_crossattn"] = [model.get_learned_conditioning([inst_prompt, flaw_prompt] if args.enable_flaw_prompt else [inst_prompt])]
        
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(next(model.parameters()).device)
            # input image embedding 
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

            uncond = {
                "c_crossattn": [torch.cat([null_token, null_token], 0) if args.enable_flaw_prompt else null_token],
                "c_concat": [torch.zeros_like(cond["c_concat"][0])]
            }

            sigmas = model_wrap.get_sigmas(args.steps if 'inpaint' not in image_path else 50)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": args.cfg_text,
                "image_cfg_scale": args.cfg_image,
                "input_image": input_image
            }

            torch.manual_seed(seed)
            _, skip_connect_hs = model.first_stage_model.encoder(input_image)
            z = torch.randn_like(cond["c_concat"][0], device='cuda:0', requires_grad=True) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = model.decode_first_stage(z, skip_connect_hs=skip_connect_hs)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

            print(f"Save images to {image_path.split('.')[0] +'.jpg'}")
            edited_image.save(os.path.join(args.outdir, f"{image_path.split('.')[0]}.jpg"))
            # input_image_pil.save(os.path.join(args.outdir, f"{image_path.split('.')[0]}_input.jpg"))
if __name__ == "__main__":
    main()
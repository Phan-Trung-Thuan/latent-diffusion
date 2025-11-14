import argparse, os, sys, glob
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import gc

import sys
sys.path.append('.')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, device_name='cpu', verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cuda:0' if device_name == 'cuda' else 'cpu')
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device_name == 'cuda':
        model.half().cuda()
    model.eval()
    return model

import torch
import numpy as np
from tqdm import tqdm

# ==================================================
# GHÉP LATENT: lấy nửa trái từ slice1, nửa phải từ slice2
# ==================================================
def splice(left, right):
    B, C, H, W = left.shape
    cut = W // 2
    return torch.cat([left[:, :, :, :cut], right[:, :, :, cut:]], dim=3)


# ==================================================
# CHẠY 1 BƯỚC DDIM
# ==================================================
@torch.no_grad()
def ddim_step(sampler, x, c, uc, t_index):
    t = torch.tensor([sampler.ddim_timesteps[t_index]], device=x.device, dtype=x.dtype)
    c = c.to(t.device)
    print(x.device, c.device, t.device, t_index.device, uc.device)
    x_prev, pred_x0 = sampler.p_sample_ddim(
        x, c, t, index=t_index,
        unconditional_guidance_scale=5.0,
        unconditional_conditioning=uc
    )
    return x_prev


# ==================================================
# TẠO SLICE ĐẦU (LẤY FULL LATENT TRAJECTORY)
# ==================================================
def generate_first_slice(sampler, model, prompt, steps, H=256, W=256):
    with torch.no_grad():
        with torch.autocast("cuda"):
            with model.ema_scope():
                c = model.get_learned_conditioning([prompt])
                uc = model.get_learned_conditioning([""])

                shape = [4, H // 8, W // 8]

                # chạy sample để lấy intermediates
                _, intermediates = sampler.sample(
                    S=steps,
                    batch_size=1,
                    shape=shape,
                    conditioning=c,
                    unconditional_conditioning=uc,
                    unconditional_guidance_scale=5.0,
                    eta=0.0,
                    verbose=False,
                    log_every_t=1
                )

                latents = intermediates["x_inter"]  # list length = steps+1
                return latents, c, uc


# ==================================================
# TẠO SLICE THỨ 2 BẰNG CƠ CHẾ: COPY LEFT + DENOISE RIGHT STEP-BY-STEP
# ==================================================
def generate_next_slice(sampler, model, prompt, prev_latents):
    steps = len(prev_latents) - 1  # same number of steps
    device = prev_latents[0].device

    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])

    # khởi tạo latent phải = noise
    noisy = torch.randn_like(prev_latents[0])

    # bước 0
    combined = splice(prev_latents[0], noisy)
    x = ddim_step(sampler, combined, c, uc, t_index=0)

    new_latents = [x]

    # các bước tiếp theo
    for t in range(1, steps):
        combined = splice(prev_latents[t], new_latents[-1])
        x = ddim_step(sampler, combined, c, uc, t_index=t)
        new_latents.append(x)

    return new_latents


# ==================================================
# DECODE SLICE CUỐI CÙNG SAU 100 STEP
# ==================================================
def decode_slice(model, final_latent):
    decoder_dtype = next(model.first_stage_model.parameters()).dtype
    final_latent = final_latent.to(device=model.device, dtype=decoder_dtype)
    decoded = model.decode_first_stage(final_latent)
    decoded = torch.clamp((decoded + 1) / 2, 0, 1)
    img = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


# ==================================================
# MAIN — TẠO NHIỀU SLICE NỐI NHAU
# ==================================================
def extend_sequence(sampler, model, prompt, n_slices=10, steps=100, H=256, W=256):

    # slice đầu tiên
    prev_latents, _, _ = generate_first_slice(sampler, model, prompt, steps, H, W)
    prev_latents = [x.clone() for x in prev_latents]

    full_img = decode_slice(model, prev_latents[-1])

    for i in range(1, n_slices):
        print(f"Generating slice {i+1}/{n_slices}")

        new_latents = generate_next_slice(sampler, model, prompt, prev_latents)
        img = decode_slice(model, new_latents[-1])

        full_img = np.concatenate([full_img, img], axis=1)
        prev_latents = new_latents  # update

    return full_img


if __name__ == "__main__":
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = 'cuda'
        torch.cuda.empty_cache()
    gc.collect()

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt", device_name=device_name)  # TODO: check path

    device = torch.device(device_name)
    model = model.to(device)

    sampler_type = 'DDIM' # ['DDIM', 'PLMS']
    if sampler_type == 'PLMS':
        ddim_eta = 0
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    outpath = 'outputs/txt2img-samples'
    os.makedirs(outpath, exist_ok=True)

    prompt = 'a beautiful landscape, 4k photo'
    H = 512
    W = 512

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
            
    img = extend_sequence(sampler, model, prompt, n_slices=10, steps=100, H=H, W=W)
    Image.fromarray(img).save("extended.png")

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")

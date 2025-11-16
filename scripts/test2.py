import torch
import numpy as np
import os
import gc
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append('.')

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


# -----------------------------
# Load model
# -----------------------------
def load_model_from_config(config, ckpt, device_name='cpu', verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cuda:0' if device_name == 'cuda' else 'cpu')
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if device_name == 'cuda':
        model.half().cuda()

    model.eval()
    return model

# --------------------------------------------------------
# 1) Create a large latent panorama for global consistency
# --------------------------------------------------------
def init_latent_panorama(model, H, W, num_slices):
    C = 4
    latent_H = H // 8
    latent_W = W // 8

    # Big panorama latent canvas
    panorama = torch.randn(
        1, C, latent_H, latent_W * num_slices,
        device=model.device,
        dtype=next(model.parameters()).dtype
    )

    return panorama


# --------------------------------------------------------
# 2) Generate each slice into the panorama using latent passing
# --------------------------------------------------------
def generate_slice_with_context(
    sampler,
    model,
    prompt,
    steps,
    H,
    W,
    left_latent_context=None,
    slice_index=0,
    num_slices=1
):
    C = 4
    latent_H = H // 8
    latent_W = W // 8

    # Build conditioning
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])

    # Prepare initial latent
    shape = (C, latent_H, latent_W)

    # Use left neighbor latent for consistency
    samples, intermediates = sampler.sample(
        S=steps,
        batch_size=1,
        shape=shape,
        conditioning=c,
        unconditional_conditioning=uc,
        unconditional_guidance_scale=5.0,
        leading_latents=left_latent_context,   # <--- PASS LATENT HERE
        eta=0.0
    )

    return samples


# --------------------------------------------------------
# 3) Latent overlap blending between adjacent slices
# --------------------------------------------------------
def blend_overlap(left, right, overlap_w):
    """
    left, right: tensors shape (1,4,H,W)
    overlap_w: number of latent columns to blend
    """
    if overlap_w <= 0:
        return left, right

    L = left.clone()
    R = right.clone()

    # Left slice's right overlap region
    L_overlap = L[:, :, :, -overlap_w:]
    # Right slice's left overlap region
    R_overlap = R[:, :, :, :overlap_w]

    # Linear blend: 0 → 1 from left to right
    alpha = torch.linspace(0, 1, overlap_w, device=L.device).view(1, 1, 1, -1)

    blended_left = (1 - alpha) * L_overlap + alpha * R_overlap
    blended_right = alpha * L_overlap + (1 - alpha) * R_overlap

    # Write back
    L[:, :, :, -overlap_w:] = blended_left
    R[:, :, :, :overlap_w] = blended_right

    return L, R


# --------------------------------------------------------
# 4) Full panorama extension pipeline (NO CLIP)
# --------------------------------------------------------
def generate_longer_panorama(
    sampler,
    model,
    prompt,
    H,
    W,
    steps,
    num_slices,
    overlap_ratio=0.25,   # 25% overlap region
):
    latent_H = H // 8
    latent_W = W // 8

    # Init panorama
    panorama = init_latent_panorama(model, H, W, num_slices)

    # Storage for slices
    slices = []
    previous_latent = None

    overlap_w = int(latent_W * overlap_ratio)

    for i in tqdm(range(num_slices), desc="Generating slices"):

        # 1) Generate slice
        slice_latent = generate_slice_with_context(
            sampler,
            model,
            prompt,
            steps,
            H,
            W,
            left_latent_context=previous_latent,   # <--- PASS LATENT HERE
            slice_index=i,
            num_slices=num_slices
        )

        # 2) Blend with previous slice in latent-space
        if i > 0:
            slices[i-1], slice_latent = blend_overlap(slices[i-1], slice_latent, overlap_w)

        # 3) Save
        slices.append(slice_latent)
        previous_latent = slice_latent

    # -------------------------------------------------------------
    #  Stitch slices into the large panorama (latent-space concate)
    # -------------------------------------------------------------
    for i, sl in enumerate(slices):
        start = i * latent_W
        end   = start + latent_W
        panorama[:, :, :, start:end] = sl

    return panorama, slices


# --------------------------------------------------------
# 5) Decode panorama into final image
# --------------------------------------------------------
def decode_panorama(model, panorama_latent):
    decoder_dtype = next(model.first_stage_model.parameters()).dtype
    panorama_latent = panorama_latent.to(model.device, dtype=decoder_dtype)

    decoded = model.decode_first_stage(panorama_latent)
    decoded = torch.clamp((decoded + 1) / 2, 0, 1)

    img = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model  = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt", device_name)
    sampler = DDIMSampler(model)

    panorama_latent = generate_longer_panorama(
        sampler, model,
        prompt="a beautiful landscape with grass, trees, river, mountains and sky, ultra wide panorama",
        num_slices=5,
        steps=200,
        H=512,
        W=512,
        overlap_ratio=0.30
    )

    final_image = decode_panorama(model, panorama_latent)
    Image.fromarray(final_image).save("panorama_output.png")

    print("Saved panorama_output.png")

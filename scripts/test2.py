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
from transformers import CLIPModel, CLIPProcessor


# -----------------------------
# Load CLIP for cross-slice conditioning
# -----------------------------

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def get_clip_embedding(image_np):
    """ encodes numpy array image to CLIP embeddings """
    image_pil = Image.fromarray(image_np)
    inputs = clip_proc(images=image_pil, return_tensors="pt").to("cuda")
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu()


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


# -----------------------------
# Decode latent slice
# -----------------------------
def decode_latent(model, latent):
    latent = latent.to(model.device, dtype=next(model.first_stage_model.parameters()).dtype)
    x = model.decode_first_stage(latent)
    x = torch.clamp((x + 1) / 2, 0, 1)
    img = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


# -----------------------------
# Generate a patch of latent
# -----------------------------
def generate_latent_patch(
        sampler, model, prompt_embed, steps, 
        H_lat, W_lat, leading_latents=None):

    samples, intermediates = sampler.sample(
        S=steps,
        batch_size=1,
        shape=(4, H_lat, W_lat),
        conditioning=prompt_embed,
        unconditional_guidance_scale=5.0,
        unconditional_conditioning=model.get_learned_conditioning([""]),
        eta=0.0,
        leading_latents=leading_latents,
        clip_ratio=0.375,
        tail_ratio=0.125,
        return_latent_t_dict=True
    )

    final_latent, leading_out = samples
    return final_latent, leading_out


# -----------------------------
# Build prompt conditioning with Cross-slice CLIP
# -----------------------------
def build_conditioning(model, text_prompt, prev_image=None, alpha=0.5):
    text_cond = model.get_learned_conditioning([text_prompt])

    if prev_image is None:
        return text_cond

    # --- CLIP embedding ---
    clip_emb = get_clip_embedding(prev_image).to(text_cond.device)

    # Align dims (broadcast)
    clip_emb = clip_emb.unsqueeze(1)

    # Mixed conditioning
    mixed = alpha * text_cond + (1 - alpha) * clip_emb
    return mixed


# -----------------------------
# MAIN: Latent Panorama Generation + Consistency tricks
# -----------------------------
def generate_latent_panorama(
        sampler, model, prompt, 
        num_slices=6, steps=60, H=512, W=512,
        overlap_ratio=0.25
    ):

    H_lat = H // 8
    W_lat = W // 8
    overlap_lat = int(W_lat * overlap_ratio)

    # Panorama latent size
    panorama = torch.zeros(1, 4, H_lat, W_lat * num_slices).half().to(model.device)

    prev_latent = None
    prev_image = None
    leading_latents = None

    for i in range(num_slices):
        print(f"Generating slice {i+1}/{num_slices}")

        # ----- Conditioning -----
        cond = build_conditioning(model, prompt, prev_image)

        # ----- Generate latent patch using DDIM -----
        latent_patch, leading_latents = generate_latent_patch(
            sampler, model, cond, steps, 
            H_lat, W_lat,
            leading_latents=leading_latents
        )

        # ----- Overlap blending -----
        if prev_latent is not None:
            patch_left = latent_patch[:, :, :, :overlap_lat]
            pano_right = panorama[:, :, :, i * W_lat - overlap_lat : i * W_lat]

            blended = 0.5 * patch_left + 0.5 * pano_right
            panorama[:, :, :, i * W_lat - overlap_lat : i * W_lat] = blended

            panorama[:, :, :, i * W_lat : (i+1)*W_lat] = latent_patch[:, :, :, overlap_lat:]
        else:
            panorama[:, :, :, :W_lat] = latent_patch

        # ----- decode preview slice for next conditioning -----
        prev_latent = latent_patch
        prev_image = decode_latent(model, latent_patch)

    return panorama


# -----------------------------
# Decode whole panorama to image
# -----------------------------
def decode_panorama(model, pano_latent):
    pano = decode_latent(model, pano_latent)
    return pano


# ===========================================================
#               RUN
# ===========================================================

if __name__ == "__main__":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model  = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt", device_name)
    sampler = DDIMSampler(model)

    panorama_latent = generate_latent_panorama(
        sampler, model,
        prompt="a beautiful landscape with grass, trees, river, mountains and sky, ultra wide panorama",
        num_slices=10,
        steps=80,
        H=512,
        W=512,
        overlap_ratio=0.30
    )

    final_image = decode_panorama(model, panorama_latent)
    Image.fromarray(final_image).save("panorama_output.png")

    print("Saved panorama_output.png")

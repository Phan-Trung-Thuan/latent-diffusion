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

# --- Helper Functions (PLACEHOLDERS, modified for the new logic) ---

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


def generate_slice(sampler, model, prompt, steps, H, W, leading_latents=None, clip_ratio=0.375, tail_ratio=0.125, return_latent_t_dict=False):
    """
    Simulates a single full-length sequence generation step, optionally using
    leading latents for coherence and returning new leading latents.
    """
    C = 4 # Latent Channels
    batch_size = 1
    
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    
    # LDM latent space size is typically 8x smaller than H, W
    latent_H = H // 8
    latent_W = W // 8
    
    samples, intermediates = sampler.sample(
        S=steps,
        batch_size=batch_size,
        shape=(C, latent_H, latent_W), 
        conditioning=c,
        unconditional_guidance_scale=5.0,
        unconditional_conditioning=uc,
        eta=0.0,
        leading_latents=leading_latents,
        clip_ratio=clip_ratio,
        tail_ratio=tail_ratio,
        return_latent_t_dict=return_latent_t_dict,
    )
    
    if return_latent_t_dict:
        samples, latent_t_to_out = samples
    decoded_output = decode_slice(model, samples)
    
    # The dictionary of intermediate latents is nested in intermediates
    # Note: DDIMSampler.sample returns (samples, intermediates), where intermediates 
    # is either the standard dict or (latent_dict, standard_dict) if return_latent_t_dict is True
    
    if return_latent_t_dict:
        # Assuming the modified DDIMSampler returns (final_result, intermediates) where 
        # final_result is (img, latent_t_to_out) if return_latent_t_dict is True
        return decoded_output, latent_t_to_out # samples[1] is the latent_t_to_out dict
    
    return decoded_output, None


# --- Refactored Sequence Extension Logic (generate_longer_by_slices) ---

def generate_longer_by_slices(sampler, model, prompt, num_slices=5, steps=100, H=256, W=256, return_slices: bool = False):
    """
    Generates a longer sequence by concatenating 'num_slices' segments, 
    maintaining coherence by passing intermediate latents.
    
    This version follows the logic:
    - Each slice is a full-length generation.
    - Slices are concatenated along the sequence dimension (assumed axis=1/Width).
    - Uses default fixed clip and tail ratios.
    """
    
    # Default parameters for latent space coherence
    # These match the values in the sample logic you provided (0.375 and 0.125)
    DEFAULT_CLIP_RATIO = 0.375 # 3/8
    DEFAULT_TAIL_RATIO = 0.125 # 1/8
    overlap_ratio = 0.5 # 4/8
    
    musics = []
    leading_latents = None

    if num_slices < 1:
        return np.array([]), []

    # --- Run 1: Generate the first slice (unconditional) ---
    print(f"Generating slice 1/{num_slices} (Initial run)")
    
    # The initial run does not use leading_latents, but it must return them for the next step.
    music, _ = generate_slice(
        sampler, model, prompt, steps, H, W,
        return_latent_t_dict=True
    )
    
    clip_point = int(music.shape[1] * (1.0 - overlap_ratio))
    musics.append(music[:, :clip_point])

    # --- Subsequent Runs (i = 2 to num_slices) ---
    for i in tqdm(range(2, num_slices + 1), desc='Generating subsequent slices'):
        print(f"Generating slice {i}/{num_slices}")
        
        # Subsequent runs use the `leading_latents` from the previous step for coherence.
        music, _ = generate_slice(
            sampler, model, prompt, steps, H, W, 
            leading_latents=leading_latents, 
            clip_ratio=DEFAULT_CLIP_RATIO, 
            tail_ratio=DEFAULT_TAIL_RATIO,
            return_latent_t_dict=True # Keep returning them for the next iteration
        )

        clip_start = int(music.shape[1] * overlap_ratio)
        musics.append(music[:, clip_start:])

    # Concatenate all generated slices (assuming concatenation along the width/sequence dimension, axis=1)
    final_sequence = np.concatenate(musics, axis=1)

    return final_sequence, musics if return_slices else final_sequence


def decode_slice(model, final_latent):
    decoder_dtype = next(model.first_stage_model.parameters()).dtype
    final_latent = final_latent.to(device=model.device, dtype=decoder_dtype)
    decoded = model.decode_first_stage(final_latent)
    decoded = torch.clamp((decoded + 1) / 2, 0, 1)
    img = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


if __name__ == '__main__':
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = 'cuda'
        torch.cuda.empty_cache()
    gc.collect()

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt", device_name=device_name)  # TODO: check path

    device = torch.device(device_name)
    model = model.to(device)

    sampler = DDIMSampler(model)

    outpath = 'outputs/txt2img-samples'
    os.makedirs(outpath, exist_ok=True)

    prompt = 'a beautiful landscape'
    H = 512
    W = 512
    steps = 200
    
    # final_img, _ = generate_slice(sampler, model, prompt, steps, H, W)
    # --- Modified function call based on slice count ---
    num_slices_to_generate = 5
    
    final_img, slices = generate_longer_by_slices(
        sampler, model, prompt, 
        num_slices=num_slices_to_generate, 
        steps=steps, 
        H=H, 
        W=W,
        return_slices=True
    )
    
    print(f"Final concatenated sequence shape: {final_img.shape}")
    # Example for saving the final sequence
    Image.fromarray(final_img).save("extended_coherent_by_slices.png")
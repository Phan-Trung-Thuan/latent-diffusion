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


# -----------------------------
# 3) Decode panorama (Unchanged)
# -----------------------------
def decode_panorama(model, panorama_latent):
    # Check if the model has a first_stage_model (VAE)
    if not hasattr(model, 'first_stage_model') or model.first_stage_model is None:
        print("Error: Model VAE (first_stage_model) not available for decoding.")
        # Return a black image as fallback
        H, W = panorama_latent.shape[2] * 8, panorama_latent.shape[3] * 8
        return np.zeros((H, W, 3), dtype=np.uint8)


    decoder_dtype = next(model.first_stage_model.parameters()).dtype
    panorama_latent = panorama_latent.to(model.device, dtype=decoder_dtype)

    decoded = model.decode_first_stage(panorama_latent)
    # Clamp and normalize from [-1, 1] to [0, 1]
    decoded = torch.clamp((decoded + 1) / 2, 0, 1)

    img = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # -------------------------------------------------------------
    #  Khởi tạo và chạy model (giữ nguyên)
    # -------------------------------------------------------------
    config_path = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    ckpt_path = "models/ldm/text2img-large/model.ckpt" # Thay đổi path này nếu cần
    
    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        print("Lỗi: Không tìm thấy file config hoặc checkpoint.")
        print(f"Kiểm tra đường dẫn: {config_path} và {ckpt_path}")
        #sys.exit(1) # Lỗi nếu không tìm thấy file
    
    config = OmegaConf.load(config_path)
    model  = load_model_from_config(config, ckpt_path, device_name)
    sampler = DDIMSampler(model)

    uc = model.get_learned_conditioning([""])
    c = model.get_learned_conditioning(["a beautiful landscape with grass, trees, mountains and sky"])

    panorama_latent = sampler.lsjd_sample(
        num_steps=100,
        tile_shape=[1, 4, 512 // 8, 512 // 8],
        num_slices=10,
        conditioning=c,
        w_swap=2,
        ref_guided_rate=0.15,
        overlap_ratio=0.25,
        unconditional_guidance_scale=5,
        unconditional_conditioning=uc
    )

    final_image = decode_panorama(model, panorama_latent)
    Image.fromarray(final_image).save("panorama_with_latent_swap_output.png")

    print("Saved panorama_with_latent_swap_output.png")
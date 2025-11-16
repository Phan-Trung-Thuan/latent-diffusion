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
# 0) Synchronization Core (LSJD Logic)
# -----------------------------
def synchronize_latents(latents_x0_list, overlap_w):
    """
    Applies Latent Swap (Averaging) to the predicted denoised estimates (x0)
    in the overlap regions across all adjacent slices.
    This is the core of Joint Diffusion for seam removal.

    Args:
        latents_x0_list (list of Tensors): List of predicted x0 for all slices.
        overlap_w (int): Number of latent columns to blend.
    """
    if overlap_w <= 0 or len(latents_x0_list) < 2:
        return latents_x0_list

    # Process all adjacent pairs
    for i in range(len(latents_x0_list) - 1):
        left_x0 = latents_x0_list[i]
        right_x0 = latents_x0_list[i+1]

        # 1. Define Overlap Regions
        L_overlap = left_x0[:, :, :, -overlap_w:]
        R_overlap = right_x0[:, :, :, :overlap_w]

        # 2. Linear Blend: 0 → 1 from left (L) to right (R)
        # alpha is the weight of the RIGHT tensor
        alpha = torch.linspace(0, 1, overlap_w, device=left_x0.device).view(1, 1, 1, -1)

        # Blended value = (Weight of Left) * Left_Value + (Weight of Right) * Right_Value
        blended_overlap = (1 - alpha) * L_overlap + alpha * R_overlap

        # 3. Write Blended Overlap back to BOTH slices
        latents_x0_list[i][:, :, :, -overlap_w:] = blended_overlap
        latents_x0_list[i+1][:, :, :, :overlap_w] = blended_overlap

    return latents_x0_list

# -----------------------------
# 2) LSJD Sampling Loop
# -----------------------------
def generate_longer_panorama_lsjd(
    sampler,
    model,
    prompt,
    H,
    W,
    steps,
    num_slices,
    overlap_ratio=0.30,
    cfg_scale=5.0
):
    latent_H = H // 8
    latent_W = W // 8
    C = 4
    device = model.device
    dtype = next(model.parameters()).dtype

    # 1. Conditioning
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    uc_full = uc
    c_full = c

    # 2. Schedule and Overlap
    # ddim_schedule is a dictionary containing ddim_timesteps, ddim_alphas, etc.
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)
    
    # KHẮC PHỤC LỖI: Đảm bảo tất cả các tham số lịch trình (schedule parameters) là PyTorch Tensors 
    # và tránh sử dụng .cpu() trên các đối tượng numpy.ndarray.
    
    # 1. Chuyển ddim_alphas_prev sang Tensor và tính sqrt_one_minus_alphas_prev.
    # Dùng torch.as_tensor để xử lý NumPy array hoặc Tensor. Thêm .float() để nhất quán kiểu.
    ddim_alphas_prev_tensor = torch.as_tensor(sampler.ddim_alphas_prev, device=device, dtype=dtype)
    ddim_sqrt_one_minus_alphas_prev = torch.sqrt(1. - ddim_alphas_prev_tensor)

    # 2. Chuyển ddim_alphas và ddim_sqrt_one_minus_alphas sang Tensor.
    ddim_alphas_tensor = torch.as_tensor(sampler.ddim_alphas, device=device, dtype=dtype)
    ddim_sqrt_one_minus_alphas_tensor = torch.as_tensor(sampler.ddim_sqrt_one_minus_alphas, device=device, dtype=dtype)

    ddim_timesteps = sampler.ddim_timesteps
    time_range = np.asarray(ddim_timesteps)[::-1]
    total_steps = ddim_timesteps.shape[0]

    overlap_w = int(latent_W * overlap_ratio)

    # 3. Initialize Latents (All slices start as pure noise)
    latents_list = []
    for _ in range(num_slices):
        noise = torch.randn(1, C, latent_H, latent_W, device=device, dtype=dtype)
        latents_list.append(noise)

    # 4. LSJD Denoising Loop
    with torch.no_grad():
        with tqdm(time_range, desc="LSJD Sampling (Joint Diffusion)") as t_iterator:
            for i, step in enumerate(t_iterator):
                ts = torch.full((1,), step, device=device, dtype=torch.long)

                all_x0_preds = []
                all_eps_preds = []

                # A) Predict Noise and Estimate x0 for ALL slices
                for x_t in latents_list:
                    # DDIM Step (Forward)
                    # --------------------
                    # 1. Predict noise (epsilon)
                    # Combine conditioning: [unconditional, conditional]
                    x_in = torch.cat([x_t] * 2)
                    t_in = torch.cat([ts] * 2)
                    c_in = torch.cat([uc_full, c_full], dim=0)

                    e_t_uncond, e_t = model.apply_model(x_in, t_in, c_in).chunk(2)

                    # 2. Classifier-Free Guidance (CFG) on epsilon
                    e_t_cfg = e_t_uncond + cfg_scale * (e_t - e_t_uncond)

                    # 3. Predict the original image latent (x0)
                    # SỬ DỤNG TENSORS ĐÃ CHUYỂN ĐỔI (Slicing [i:i+1] để giữ nguyên là Tensor):
                    a_t = ddim_alphas_tensor[i:i+1] 
                    sqrt_one_minus_at = ddim_sqrt_one_minus_alphas_tensor[i:i+1]
                    
                    pred_x0 = (x_t - sqrt_one_minus_at * e_t_cfg) / torch.sqrt(a_t)

                    all_x0_preds.append(pred_x0)
                    all_eps_preds.append(e_t_cfg)

                # B) Latent Swap (Synchronization) - Apply on x0
                # ------------------------------------------------
                all_x0_preds_sync = synchronize_latents(all_x0_preds, overlap_w)

                # C) Calculate next latent x_{t-1} using synchronized x0
                # --------------------------------------------------------
                for j in range(num_slices):
                    pred_x0 = all_x0_preds_sync[j]
                    e_t_cfg = all_eps_preds[j]
                    
                    # DDIM Step (Reverse)
                    # --------------------
                    # SỬ DỤNG TENSORS ĐÃ CHUYỂN ĐỔI (Slicing [i:i+1] để giữ nguyên là Tensor):
                    a_prev = ddim_alphas_prev_tensor[i:i+1]
                    sqrt_one_minus_at_prev = ddim_sqrt_one_minus_alphas_prev[i:i+1] 

                    # x_{t-1} = sqrt(alpha_t-1) * pred_x0 + sqrt(1 - alpha_t-1) * e_t_cfg
                    dir_xt = sqrt_one_minus_at_prev * e_t_cfg
                    x_prev = torch.sqrt(a_prev) * pred_x0 + dir_xt

                    latents_list[j] = x_prev

    # 5. Stitch Final Latents
    total_latent_W = latent_W * num_slices - overlap_w * (num_slices - 1)
    
    # Create the final panorama latent tensor
    panorama_latent = torch.zeros(1, C, latent_H, total_latent_W, device=device, dtype=dtype)
    current_w = 0

    for i, sl in enumerate(latents_list):
        if i == 0:
            # First slice is fully included
            panorama_latent[:, :, :, :latent_W] = sl
            current_w += latent_W
        else:
            # Append remaining part of slice after overlap
            start = current_w - overlap_w
            end_append = latent_W - overlap_w
            
            # The overlap region is already synchronized in the previous slice's last part
            panorama_latent[:, :, :, start:start + end_append] = sl[:, :, :, overlap_w:]
            current_w += end_append
    
    # Return the single stitched latent tensor
    return panorama_latent, latents_list


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

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model  = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt", device_name)
    sampler = DDIMSampler(model)

    # Parameters for the long-form generation (e.g., 5 slices of 512x512)
    NUM_SLICES = 5
    IMAGE_H = 512
    IMAGE_W = 512
    STEPS = 50 # Reduced steps for faster testing
    OVERLAP_RATIO = 0.30 # 30% latent overlap

    print(f"Bắt đầu Tạo Panorama LSJD với {NUM_SLICES} lát cắt ({IMAGE_H}x{IMAGE_W})")
    print(f"Tỉ lệ chồng lấn: {OVERLAP_RATIO:.0%}")
    
    # The latent width of each slice
    latent_W = IMAGE_W // 8
    # The width of the overlap region in latent space
    overlap_w = int(latent_W * OVERLAP_RATIO)
    print(f"Chiều rộng Latent lát cắt: {latent_W} | Chiều rộng Latent chồng lấn: {overlap_w}")

    panorama_latent, slices = generate_longer_panorama_lsjd(
        sampler, model,
        prompt="a majestic ultra wide panorama of a medieval castle on a rocky cliff overlooking a stormy ocean, dramatic lighting, highly detailed, fantasy art",
        num_slices=NUM_SLICES,
        steps=STEPS,
        H=IMAGE_H,
        W=IMAGE_W,
        overlap_ratio=OVERLAP_RATIO
    )

    final_image = decode_panorama(model, panorama_latent)
    output_filename = "lsjd_panorama_output.png"
    Image.fromarray(final_image).save(output_filename)

    print(f"\n--- Quá trình hoàn tất ---")
    print(f"Panorama LSJD đã được lưu thành {output_filename}")
    # Calculate final pixel resolution
    final_W = final_image.shape[1]
    final_H = final_image.shape[0]
    print(f"Độ phân giải ảnh cuối cùng: {final_W}x{final_H}")

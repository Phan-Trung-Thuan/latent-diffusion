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
# 1) Create a large latent panorama for global consistency (GIỮ NGUYÊN)
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
# 2) Generate each slice with Latent Context (ĐÃ SỬA ĐỔI)
# --------------------------------------------------------
def generate_slice_with_context(
    sampler,
    model,
    prompt,
    steps,
    H,
    W,
    leading_latents_dict=None, # Dictionary latent trung gian từ slice trước
    clip_ratio=0.375, # <--- Nhận tham số động
    tail_ratio=0.125,
    slice_index=0,
    num_slices=1
):
    C = 4
    latent_H = H // 8
    latent_W = W // 8
    device = model.device

    # Build conditioning
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])

    # Prepare initial latent
    shape = (C, latent_H, latent_W)

    # Lấy mẫu. Cần return_latent_t_dict=True để lấy latent trung gian cho slice tiếp theo
    # Lưu ý: sampler.sample đã được sửa để trả về (final_latent, intermediates, latent_t_dict)
    final_latent, intermediates, latent_t_dict = sampler.sample(
        S=steps,
        batch_size=1,
        shape=shape,
        conditioning=c,
        unconditional_conditioning=uc,
        unconditional_guidance_scale=5.0,
        leading_latents=leading_latents_dict, # <-- TRUYỀN DICTIONARY LATENT TỪ SLICE TRƯỚC VÀO SAMPLER
        eta=0.0,
        return_latent_t_dict=True, # <-- Bật cờ để lấy latent trung gian
        clip_ratio=clip_ratio, # <--- TRUYỀN THAM SỐ ĐỘNG
        tail_ratio=tail_ratio
    )

    # Chỉ trả về latent cuối cùng (x_0) và dictionary latent trung gian (x_t)
    return final_latent, latent_t_dict


# --------------------------------------------------------
# 3) Latent overlap blending between adjacent slices (GIỮ NGUYÊN)
# --------------------------------------------------------
def blend_overlap(left, right, overlap_w):
    """
    Sửa lỗi: Chỉ tính toán giá trị blend một lần và gán lại cho cả hai slice.
    """
    if overlap_w <= 0:
        return left, right

    L = left.clone()
    R = right.clone()
    device = L.device

    L_overlap = L[:, :, :, -overlap_w:]
    R_overlap = R[:, :, :, :overlap_w]

    # Linear blend: alpha chạy từ 0 (cực trái L) đến 1 (cực phải R)
    alpha = torch.linspace(0, 1, overlap_w, device=device).view(1, 1, 1, -1)

    # Tính toán giá trị blend duy nhất
    blended_value = (1 - alpha) * L_overlap + alpha * R_overlap 

    # Ghi đè giá trị blend vào vùng overlap của cả hai slice
    L[:, :, :, -overlap_w:] = blended_value
    R[:, :, :, :overlap_w] = blended_value

    return L, R


# --------------------------------------------------------
# 4) Full panorama extension pipeline (ĐÃ SỬA ĐỔI)
# --------------------------------------------------------
def generate_longer_panorama(
    sampler,
    model,
    prompt,
    H,
    W,
    steps,
    num_slices,
    overlap_ratio=0.35, # <--- Tăng nhẹ Overlap để cải thiện độ mượt
    tail_ratio=0.1,    # <--- Tham số tinh chỉnh Latent Swap
):
    latent_H = H // 8
    latent_W = W // 8
    
    slices = []
    previous_latent_dict = None
    final_panorama = None
    overlap_w = int(latent_W * overlap_ratio)

    # ĐỒNG BỘ HÓA: clip_ratio phải bằng overlap_ratio để Latent Swap dẫn hướng vùng blend
    latent_clip_ratio = overlap_ratio

    for i in tqdm(range(num_slices), desc="Generating slices"):

        # 1) Generate slice và LƯU latent trung gian của nó
        slice_latent, current_latent_dict = generate_slice_with_context(
            sampler,
            model,
            prompt,
            steps,
            H,
            W,
            leading_latents_dict=previous_latent_dict,
            clip_ratio=latent_clip_ratio, # <-- Dùng overlap_ratio
            tail_ratio=tail_ratio,       # <-- Dùng tail_ratio đã tinh chỉnh
            slice_index=i,
            num_slices=num_slices
        )
        
        # 2) Blend và Nối (CHỈ THỰC HIỆN 1 LẦN)
        if i > 0:
            # a. Blend x_0 giữa slice trước (slices[i-1]) và slice hiện tại (slice_latent)
            slices[i-1], slice_latent = blend_overlap(slices[i-1], slice_latent, overlap_w)
            
            # b. Nối phần KHÔNG chồng lấn của slice hiện tại vào panorama
            # Lấy phần KHÔNG BỊ TRÙNG LẶP của slice hiện tại (từ vị trí overlap_w đến hết)
            non_overlap_part = slice_latent[:, :, :, overlap_w:]
            final_panorama = torch.cat([final_panorama, non_overlap_part], dim=-1)
        else:
            # Slice đầu tiên: Khởi tạo panorama bằng phần không overlap (chờ nối)
            final_panorama = slice_latent[:, :, :, :-overlap_w] # <-- Lấy phần đầu, trừ phần overlap cuối
            
        # 3) Lưu latent cuối cùng và cập nhật dictionary latent context cho slice tiếp theo
        slices.append(slice_latent)
        previous_latent_dict = current_latent_dict # <-- CẬP NHẬT CONTEXT

    # LOẠI BỎ logic nối chuỗi lặp lại ở đây.
    
    return final_panorama, slices

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

    panorama_latent, slices = generate_longer_panorama(
        sampler, model,
        prompt="a beautiful landscape with grass, trees, river, mountains and sky",
        num_slices=4, # Đặt số slice nhỏ để test
        steps=100, # Giảm bước để test nhanh
        H=512,
        W=512,
        overlap_ratio=0.35,
        tail_ratio=0.0
    )

    final_image = decode_panorama(model, panorama_latent)
    Image.fromarray(final_image).save("panorama_with_latent_swap_output.png")

    print("Saved panorama_with_latent_swap_output.png")
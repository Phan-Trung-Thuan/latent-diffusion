import torch
import numpy as np
import os
import gc
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append('.')

# Giả định DDIMSampler đã được mở rộng để bao gồm lsjd_sample và các hàm phụ trợ
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
# Decode panorama (Unchanged)
# -----------------------------
def decode_panorama(model, panorama_latent):
    # Check if the model has a first_stage_model (VAE)
    if not hasattr(model, 'first_stage_model') or model.first_stage_model is None:
        print("Error: Model VAE (first_stage_model) not available for decoding.")
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
    # Khởi tạo và chạy model (giữ nguyên)
    # -------------------------------------------------------------
    config_path = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    ckpt_path = "models/ldm/text2img-large/model.ckpt"
    
    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        print("Lỗi: Không tìm thấy file config hoặc checkpoint.")
        print(f"Kiểm tra đường dẫn: {config_path} và {ckpt_path}")
        # sys.exit(1)
        
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, device_name)
    # Khởi tạo sampler. Giả định sampler đã có phương thức lsjd_sample
    sampler = DDIMSampler(model) 

    # --- SỬA ĐỔI ĐỂ DÙNG MULTI-PROMPT ---
    
    # 1. Định nghĩa các tham số cơ bản
    W_TILE = 512
    OVERLAP_RATIO = 0.25
    NUM_SLICES_TOTAL = 10 # Số slice dự kiến
    
    # 2. Tính toán kích thước latent (latent_w = pixel_w // 8)
    latent_h = W_TILE // 8 # 64
    latent_w = W_TILE // 8 # 64
    
    # Kích thước Panorama tính toán: 496 (từ các bước tính toán trước)
    stride = int(latent_w * (1 - OVERLAP_RATIO)) # 48
    PANORAMA_W_LATENT = latent_w + (NUM_SLICES_TOTAL - 1) * stride # 64 + 9 * 48 = 496
    
    print(f"Kích thước Panorama Latent dự kiến: {latent_h}x{PANORAMA_W_LATENT}")

    # 3. Tạo Conditioning Vectors (Cần phải nằm trong phần LSJD Sampler)
    # Chúng ta định nghĩa các Prompt và vị trí, sau đó LSJD Sampler sẽ tự gọi model.get_learned_conditioning
    
    # Định nghĩa ba vùng prompt:
    # Vùng 1: Cỏ, Núi, Trời (0 -> 2/3)
    # Vùng 2: Núi, Trời, Hồ (1/3 -> 3/3)
    # Vùng 3: Hồ, Bãi biển, Hoàng hôn (2/3 -> cuối)
    
    W_DIV = PANORAMA_W_LATENT // 3 # Khoảng 165
    
    # Dictionary chứa các prompt và khu vực ảnh hưởng
    multi_prompts_data = {
        'left_scene': {
            'text': "A lush green meadow with tall grass, rolling mountains, and a clear blue sky, photorealistic.",
            'W_start': 0, 'W_end': W_DIV * 2, 'guidance': 7.0 
        },
        'mid_scene': {
            'text': "Snowy mountain peaks and a reflective blue lake, hyperdetailed, mystical.",
            'W_start': W_DIV, 'W_end': PANORAMA_W_LATENT, 'guidance': 5.5
        },
        'right_scene': {
            'text': "A warm tropical beach at sunset with purple and orange colors, calm ocean waves, cinematic lighting.",
            'W_start': W_DIV * 2 - stride, 'W_end': PANORAMA_W_LATENT, 'guidance': 8.0
        }
    }
    
    # 4. Chạy LSJD Multi-Prompt Sampler
    # Giả định sampler.lsjd_sample đã được sửa đổi để nhận multi_prompts
    print("\nBắt đầu lấy mẫu LSJD Multi-Prompt...")
    panorama_latent = sampler.lsjd_multi_sample(
        num_steps=100, # Giảm bước để test nhanh
        tile_shape=[1, 4, latent_h, latent_w],
        multi_prompts=multi_prompts_data,
        overlap_ratio=OVERLAP_RATIO,
        w_swap=4,
        ref_guided_rate=0.4, # Áp dụng Reference-Guided Swap trong 40% bước đầu
        eta=0.0, # Lấy mẫu xác định (Deterministic)
    )

    # 5. Giải mã và Lưu ảnh
    final_image = decode_panorama(model, panorama_latent)
    Image.fromarray(final_image).save("panorama_multi_prompt_output.png")

    print(f"\n✅ Đã lưu kết quả đa prompt vào: panorama_multi_prompt_output.png (Kích thước: {final_image.shape[1]}x{final_image.shape[0]})")
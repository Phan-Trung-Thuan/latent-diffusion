"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    # @torch.no_grad()
    # def sample(self,
    #            S,
    #            batch_size,
    #            shape,
    #            conditioning=None,
    #            callback=None,
    #            normals_sequence=None,
    #            img_callback=None,
    #            quantize_x0=False,
    #            eta=0.,
    #            mask=None,
    #            x0=None,
    #            temperature=1.,
    #            noise_dropout=0.,
    #            score_corrector=None,
    #            corrector_kwargs=None,
    #            verbose=True,
    #            x_T=None,
    #            log_every_t=100,
    #            unconditional_guidance_scale=1.,
    #            unconditional_conditioning=None,
    #            # New parameters
    #            leading_latents=None,
    #            clip_ratio=0.375,
    #            tail_ratio=0.125,
    #            return_latent_t_dict=False,
    #            # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
    #            **kwargs
    #            ):
    #     if conditioning is not None:
    #         if isinstance(conditioning, dict):
    #             cbs = conditioning[list(conditioning.keys())[0]].shape[0]
    #             if cbs != batch_size:
    #                 print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
    #         else:
    #             if conditioning.shape[0] != batch_size:
    #                 print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

    #     self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
    #     # sampling
    #     C, H, W = shape
    #     size = (batch_size, C, H, W)
    #     print(f'Data shape for DDIM sampling is {size}, eta {eta}')

    #     samples, intermediates = self.ddim_sampling(conditioning, size,
    #                                                  callback=callback,
    #                                                  img_callback=img_callback,
    #                                                  quantize_denoised=quantize_x0,
    #                                                  mask=mask, x0=x0,
    #                                                  ddim_use_original_steps=False,
    #                                                  noise_dropout=noise_dropout,
    #                                                  temperature=temperature,
    #                                                  score_corrector=score_corrector,
    #                                                  corrector_kwargs=corrector_kwargs,
    #                                                  x_T=x_T,
    #                                                  log_every_t=log_every_t,
    #                                                  unconditional_guidance_scale=unconditional_guidance_scale,
    #                                                  unconditional_conditioning=unconditional_conditioning,
    #                                                  # Pass new parameters
    #                                                  leading_latents=leading_latents,
    #                                                  clip_ratio=clip_ratio,
    #                                                  tail_ratio=tail_ratio,
    #                                                  return_latent_t_dict=return_latent_t_dict,
    #                                                  )
        
    #     if return_latent_t_dict:
    #         final_img, latent_t_dict = samples
    #         return final_img, intermediates, latent_t_dict
    #     else:
    #         return samples, intermediates

    # @torch.no_grad()
    # def ddim_sampling(self, cond, shape,
    #                   x_T=None, ddim_use_original_steps=False,
    #                   callback=None, timesteps=None, quantize_denoised=False,
    #                   mask=None, x0=None, img_callback=None, log_every_t=100,
    #                   temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
    #                   unconditional_guidance_scale=1., unconditional_conditioning=None,
    #                   # New parameters
    #                   leading_latents=None, clip_ratio=0.375, tail_ratio=0.125, return_latent_t_dict=False,
    #                   ):
    #     device = self.model.betas.device
    #     b = shape[0]
    #     if x_T is None:
    #         img = torch.randn(shape, device=device)
    #     else:
    #         img = x_T

    #     if timesteps is None:
    #         timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
    #     elif timesteps is not None and not ddim_use_original_steps:
    #         subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
    #         timesteps = self.ddim_timesteps[:subset_end]

    #     intermediates = {'x_inter': [img], 'pred_x0': [img]}
    #     # New dictionary for collecting latents at each step (if requested)
    #     latent_t_to_out = {}

    #     time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
    #     total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
    #     print(f"Running DDIM Sampling with {total_steps} timesteps")

    #     iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

    #     for i, step in enumerate(iterator):
    #         index = total_steps - i - 1
    #         ts = torch.full((b,), step, device=device, dtype=torch.long)

    #         if mask is not None:
    #             assert x0 is not None
    #             img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
    #             img = img_orig * mask + (1. - mask) * img

    #         outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
    #                                       quantize_denoised=quantize_denoised, temperature=temperature,
    #                                       noise_dropout=noise_dropout, score_corrector=score_corrector,
    #                                       corrector_kwargs=corrector_kwargs,
    #                                       unconditional_guidance_scale=unconditional_guidance_scale,
    #                                       unconditional_conditioning=unconditional_conditioning)
    #         img, pred_x0 = outs

    #         # --- Start of New Latent Manipulation Logic ---
    #         if leading_latents is not None and not ddim_use_original_steps:
    #             # The 'step' variable here is the DDPM timestep for the *current* step 't'.
    #             # The 'img' is x_{t-1}, the result of the reverse step.
    #             # Assuming leading_latents is keyed by the *DDPM* timestep 't'
    #             t_item = int(step) # The current DDPM timestep t
                
    #             # Check if the current timestep is a key in the provided latents dictionary
    #             if t_item in leading_latents:
    #                 copy_start = int(img.shape[3] * clip_ratio)
    #                 tail_length = int(img.shape[3] * tail_ratio)
    #                 ref_latents = leading_latents[t_item]
                    
    #                 # Implementation of the trick:
    #                 # Overwrite the start of the newly generated latent (img) with a segment
    #                 # from the reference latent (ref_latents).
    #                 # The dimensions are typically [batch, channels, H, W] or similar.
    #                 # Assuming the "clip/tail" manipulation is on the third dimension (e.g., width or sequence length).
    #                 # latents[:, :, :copy_start] = ref_latents[:, :, - copy_start - tail_length: - tail_length]
    #                 # This logic seems designed for sequence-like latents (e.g., music generation) where the
    #                 # sequence flows along the 3rd dimension.
                    
    #                 # A more standard spatial trick might be:
    #                 # img[:, :, :copy_start] = ref_latents[:, :, :copy_start]
                    
    #                 # Using the provided complex slicing logic from the example function:
    #                 try:
    #                     img[:, :, :, :copy_start] = ref_latents[:, :, :, - copy_start - tail_length: - tail_length].to(img.device)
    #                 except Exception as e:
    #                     print(f"Warning: Latent manipulation failed at timestep {t_item}. Error: {e}")
    #         # --- End of New Latent Manipulation Logic ---

    #         if return_latent_t_dict:
    #             # Store the latent (x_{t-1}) after the manipulation, keyed by the current DDPM timestep 't'.
    #             latent_t_to_out[int(step)] = img.clone().detach()

    #         if callback: callback(i)
    #         if img_callback: img_callback(pred_x0, i)

    #         if index % log_every_t == 0 or index == total_steps - 1:
    #             intermediates['x_inter'].append(img)
    #             intermediates['pred_x0'].append(pred_x0)

    #     # Return both the final image and a dictionary containing the intermediate latents if requested.
    #     final_result = (img, latent_t_to_out) if return_latent_t_dict else img
    #     return final_result, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

        
    def _left(self, x, overlap_ratio):
        _, _, _, w = x.shape
        overlap_w = int(w * overlap_ratio)
        return x[..., :overlap_w]
    

    def _right(self, x, overlap_ratio):
        _, _, _, w = x.shape
        overlap_w = int(w * overlap_ratio)
        return x[..., -overlap_w:]
    

    def _mid(self, x, overlap_ratio):
        _, _, _, w = x.shape
        overlap_w = int(w * overlap_ratio)
        return x[..., overlap_w:-overlap_w]

   
    def _swap(self, x1: torch.Tensor, x2: torch.Tensor, is_horizontal: bool = True, w_swap: int = 4) -> torch.Tensor:
        """
        Hàm Swap linh hoạt: X_new = W_swap * X1 + (1 - W_swap) * X2
        Tạo pattern 0/1 theo khối dọc theo chiều ngang (W) hoặc chiều dọc (H).
        
        LƯU Ý: w_swap không được cố định, đã được thêm làm tham số.
        """
        if w_swap < 1: 
            w_swap = 1
            
        B, C, H, W = x1.shape
        device = x1.device

        # 1. XÁC ĐỊNH CHIỀU CẦN SWAP (L) và SHAPE TARGET
        if not is_horizontal:
            L = H # Swap theo chiều dọc (Height)
            target_shape = (1, 1, H, 1)
        else: # is_horizontal == True
            L = W # Swap theo chiều ngang (Width)
            target_shape = (1, 1, 1, W)

        # 2. TÍNH VECTOR TRỌNG SỐ 1D (W_SWAP_VECTOR)
        i_indices = torch.arange(1, L + 1, device=device).float()
        
        # pattern_val = floor((i - 1) / w_swap)
        pattern_val = torch.floor((i_indices - 1) / w_swap)
        
        # v_m_pattern: [0, 0, 1, 1, 0, 0, ...]
        v_m_pattern = 0.5 * (1.0 - ((-1.0) ** pattern_val))
        
        # W_swap_vector: [1, 1, 0, 0, 1, 1, ...] (Đảm bảo khối đầu tiên lấy X1)
        W_swap_vector = 1.0 - v_m_pattern

        # 3. MỞ RỘNG (BROADCAST) W_SWAP
        W_swap = W_swap_vector.view(target_shape).expand(B, C, H, W)
        
        # 4. THỰC HIỆN PHÉP TRỘN/HOÁN ĐỔI
        W_swap_complement = 1.0 - W_swap
        X_new = W_swap * x1 + W_swap_complement * x2
        
        return X_new

    @torch.no_grad()
    def lsjd_sample(self,
                    num_steps: int,
                    tile_shape: tuple,
                    num_slices: int,
                    conditioning: torch.Tensor,
                    overlap_ratio: float, 
                    w_swap: int = 4,
                    ref_guided_rate: float = 0.3,
                    unconditional_guidance_scale=1.,
                    unconditional_conditioning=None,
                    eta: float = 0.,
                    verbose: bool = False
                    ):
        
        batch_size, _, _, w = tile_shape

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")


        self.make_schedule(ddim_num_steps=num_steps, ddim_eta=eta, verbose=verbose)
        total_steps = self.ddim_timesteps.shape[0]
        time_range = np.flip(self.ddim_timesteps)

        # sampling
        device = conditioning.device
        panorama_shape = list(tile_shape)
        panorama_shape[-1] = w + (num_slices - 1) * int(w * (1 - overlap_ratio))
        J = torch.randn(panorama_shape).to(device)

        # Lập qua mỗi slice theo overlap_ratio, ở mỗi bước lấy slice đó ra từ J, 
        slice_list = [J[..., i:i+w] for i in range(0, panorama_shape[-1], int(w * (1 - overlap_ratio)))][:num_slices]

        # Xác định bước dừng cho Reference-Guided Swap
        ref_guided_stop_step = int(total_steps * ref_guided_rate)

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for n, step in enumerate(iterator):
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            index = total_steps - n - 1

            # Khử nhiễu từng slice
            for i in range(len(slice_list)):
                slice_list[i], _ = self.p_sample_ddim(slice_list[i], conditioning, ts, index,
                                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                                      unconditional_conditioning=unconditional_conditioning)
            
            # 1. Xử lý Self-Loop Latent Swap (Ở VÙNG CHỒNG LẤN)
            for i in range(1, len(slice_list)):
                right_prev_i = self._right(slice_list[i - 1], overlap_ratio)
                left_i = self._left(slice_list[i], overlap_ratio)
                
                # SỬA LỖI: Truyền w_swap vào hàm _swap
                swap_zone = self._swap(left_i, right_prev_i, is_horizontal=True, w_swap=w_swap)

                overlap_w = int(w * overlap_ratio)
                # Cập nhật hai chiều
                slice_list[i - 1][..., -overlap_w:] = swap_zone
                slice_list[i][..., :overlap_w] = swap_zone

            # 2. Xử lý Reference-Guided Latent Swap (Ở VÙNG KHÔNG CHỒNG LẤN)
            if n < ref_guided_stop_step: # Chỉ thực hiện trong các bước khử nhiễu ban đầu
                # Lát cắt đầu tiên (slice_list[0]) được dùng làm tham chiếu (X_ref)
                for i in range(1, len(slice_list)):
                    # Giả định _mid trả về phần giữa của lát cắt
                    mid_ref = self._mid(slice_list[0], overlap_ratio)
                    mid_i = self._mid(slice_list[i], overlap_ratio)
                    
                    # SỬA LỖI: Truyền w_swap. 
                    # Giả định is_horizontal=True vì panorama là ngang.
                    swap_zone = self._swap(mid_ref, mid_i, is_horizontal=True, w_swap=w_swap) 

                    # Cập nhật phần giữa của slice i (Hoán đổi một chiều: Ref -> Slice i)
                    overlap_w = int(w * overlap_ratio)
                    slice_list[i][..., overlap_w:-overlap_w] = swap_zone

        # ... (Phần ghép slice cuối cùng giữ nguyên) ...
        
        for i, slice in zip(range(0, panorama_shape[-1], int(w * (1 - overlap_ratio))), slice_list):
            J[..., i:i+w] = slice

        return J
    
    @torch.no_grad()
    def lsjd_multi_sample(self,
                    num_steps: int,
                    tile_shape: tuple,
                    multi_prompts: dict,  # Ví dụ: {'prompt_1': {'W_start': 0, 'W_end': 512, 'text': 'grass', 'guidance': 7.5}, ...}
                    overlap_ratio: float, 
                    w_swap: int = 4,
                    ref_guided_rate: float = 0.3,
                    eta: float = 0.,
                    verbose: bool = False
                    ):
        
        # 0. Thiết lập và Khởi tạo Conditioning
        
        # Lấy unconditional conditioning (uc) chung cho toàn bộ panorama
        uncond_prompt = [""]
        # Tên thuộc tính có thể khác nhau (ví dụ: self.model.cond_stage_model)
        uc = self.model.get_learned_conditioning(uncond_prompt) 

        # 1. Tiền xử lý và Lưu trữ Conditioning Vectors
        # Dictionary mới lưu trữ các vector conditioning đã học (c) cho từng prompt
        conditioning_vectors = {}
        
        for key, data in multi_prompts.items():
            prompt = data['text']
            # Lấy conditional conditioning (c) cho từng prompt
            c = self.model.get_learned_conditioning([prompt])
            
            # Lưu trữ tất cả dữ liệu cần thiết: c, uc, guidance scale, và vị trí
            conditioning_vectors[key] = {
                'c': c,
                'uc': uc, # Sử dụng uc chung
                'guidance': data.get('guidance', 7.5),
                'W_start': data['W_start'],
                'W_end': data['W_end']
            }
            
        # --- Phần còn lại của Logic Khởi tạo (Tính Panorama Width, J, slice_list) ---
        
        batch_size, _, _, w_slice = tile_shape
        self.make_schedule(ddim_num_steps=num_steps, ddim_eta=eta, verbose=verbose)
        total_steps = self.ddim_timesteps.shape[0]
        time_range = np.flip(self.ddim_timesteps)
        device = uc.device
        
        max_w_end = max(data['W_end'] for data in multi_prompts.values())
        stride = int(w_slice * (1 - overlap_ratio))
        
        panorama_width = max_w_end 
        num_slices = (panorama_width - w_slice) // stride + 1

        panorama_shape = list(tile_shape)
        panorama_shape[-1] = panorama_width 
        J = torch.randn(panorama_shape).to(device)

        start_points = [i * stride for i in range(num_slices)]
        slice_list = [J[..., start_idx:start_idx+w_slice] for start_idx in start_points]

        ref_guided_stop_step = int(total_steps * ref_guided_rate)

        # 2. Vòng lặp Lấy mẫu DDIM
        iterator = tqdm(time_range, desc='LSJD Multi-Prompt Sampler', total=total_steps)
        for n, step in enumerate(iterator):
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            index = total_steps - n - 1

            # --- Xử lý Khử nhiễu Đa Điều kiện ---
            for i in range(len(slice_list)):
                start_idx = start_points[i]
                end_idx = start_idx + w_slice
                
                # 2a. Tìm Bộ Điều kiện cho Slice hiện tại (i)
                # Áp dụng giải pháp trộn/lấy trung bình nếu có nhiều prompt chồng lấn
                active_cond_data = [] 
                
                for data in conditioning_vectors.values():
                    if start_idx < data['W_end'] and end_idx > data['W_start']:
                        active_cond_data.append(data)

                if active_cond_data:
                    # 2b. TRỘN HOẶC CHỌN Điều kiện
                    
                    # GIẢI PHÁP ĐƠN GIẢN: CHỈ LẤY PROMPT ĐẦU TIÊN
                    data = active_cond_data[0]
                    active_conditioning = data['c']
                    active_uncond = data['uc']
                    active_guidance = data['guidance']
                    
                    # (GIẢI PHÁP NÂNG CAO: Trộn các vector c và uc, nhưng cần cân nhắc vùng chồng lấn)
                    # ... (Logic phức tạp hơn nếu cần) ...
                    
                else:
                    # Nếu không có prompt nào khớp
                    active_conditioning = uc
                    active_uncond = uc
                    active_guidance = 1.0 # Guidance 1.0 = không điều kiện
                
                # 2c. Khử nhiễu (Dùng điều kiện đã tìm thấy)
                slice_list[i], _ = self.p_sample_ddim(slice_list[i], active_conditioning, ts, index,
                                                    unconditional_guidance_scale=active_guidance,
                                                    unconditional_conditioning=active_uncond)
            
            # --- 3. Self-Loop Latent Swap (Giữ nguyên) ---
            for i in range(num_slices - 1):
                right_prev_i = self._right(slice_list[i], overlap_ratio)
                left_i_plus_1 = self._left(slice_list[i + 1], overlap_ratio)
                swap_zone = self._swap(left_i_plus_1, right_prev_i, is_horizontal=True, w_swap=w_swap)
                
                overlap_w = int(w_slice * overlap_ratio)
                slice_list[i][..., -overlap_w:] = swap_zone
                slice_list[i + 1][..., :overlap_w] = swap_zone

            # --- 4. Reference-Guided Latent Swap (Giữ nguyên) ---
            if n < ref_guided_stop_step:
                for i in range(1, num_slices):
                    mid_ref = self._mid(slice_list[0], overlap_ratio)
                    mid_i = self._mid(slice_list[i], overlap_ratio)
                    swap_zone = self._swap(mid_ref, mid_i, is_horizontal=True, w_swap=w_swap) 

                    overlap_w = int(w_slice * overlap_ratio)
                    mid_center = w_slice // 2
                    start_w = mid_center - overlap_w // 2
                    end_w = mid_center + overlap_w // 2
                    
                    slice_list[i][..., start_w:end_w] = swap_zone

            # 5. Hợp nhất và DDIM Step (Giữ nguyên logic ghép slice)
            J_pred = torch.zeros_like(J) 
            current_pos = 0
            for slice_x0 in slice_list:
                start_w = current_pos
                J_pred[..., start_w:start_w+w_slice] = slice_x0
                current_pos += stride
                
            # ... (Thêm DDIM step nếu cần: J = self._ddim_step_forward(J_pred, J, index)) ...
            
            J = J_pred # Tạm thời gán J_pred cho J để tiếp tục vòng lặp

        return J
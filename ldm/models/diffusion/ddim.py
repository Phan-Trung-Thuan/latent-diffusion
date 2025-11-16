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

   
    def _swap(self, x1, x2, slice_index: int, w_swap: int = 1) -> torch.Tensor:
        """
        Hàm Swap: X_new = W_swap * X1 + (1 - W_swap) * X2 theo Eq. 10.
        """
        device = x1.device
        i = slice_index + 1
        
        if w_swap == 0: w_swap = 1 
        pattern_val = (i - 1) // w_swap 
        v_m_scalar = 0.5 * (1 - ((-1) ** pattern_val))
        
        W_swap = torch.full_like(x1, v_m_scalar).to(device)
        W_swap_complement = 1.0 - W_swap
        
        # print(slice_index, W_swap.shape, x1.shape, W_swap_complement.shape, x2.shape)
        X_new = W_swap * x1 + W_swap_complement * x2
        return X_new

    @torch.no_grad()
    def lsjd_sample(self,
                    num_steps: int,
                    tile_shape: tuple,
                    num_slices: int,
                    conditioning: torch.Tensor,
                    overlap_ratio: float, 
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

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for n, step in enumerate(iterator):
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            index = total_steps - n - 1

            # Khử nhiễu từng slice
            for i in range(len(slice_list)):
                slice_list[i], _ = self.p_sample_ddim(slice_list[i], conditioning, ts, index,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            
            # Xử lý Self-Loop Latent Swap
            for i in range(1, len(slice_list)):
                right_prev_i = self._right(slice_list[i - 1], overlap_ratio)
                left_i = self._left(slice_list[i], overlap_ratio)
                swap_zone = self._swap(left_i, right_prev_i, i-1)

                overlap_w = int(w * overlap_ratio)
                # Cập nhật phần bên phải của slice i-1
                slice_list[i - 1][..., -overlap_w:] = swap_zone
                # Cập nhật phần bên trái của slice i
                slice_list[i][..., :overlap_w] = swap_zone

            # Xử lý Reference-Guided Latent Swap
            if n <= total_steps // 2:
                for i in range(1, len(slice_list)):
                    mid_0 = self._mid(slice_list[0], overlap_ratio)
                    mid_i = self._mid(slice_list[i], overlap_ratio)
                    swap_zone = self._swap(mid_0, mid_i, i)

                    overlap_w = int(w * overlap_ratio)
                    # Cập nhật phần giữa của slice i
                    slice_list[i][..., overlap_w:-overlap_w] = swap_zone

        for i, slice in zip(range(0, panorama_shape[-1], int(w * (1 - overlap_ratio))), slice_list):
            J[..., i:i+w] = slice

        return J
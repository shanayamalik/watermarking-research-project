import torch

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks

from transformers import CLIPTextModel, CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer

from typing import Union, List, Optional, Dict, Any, Callable
import numpy as np
from PIL import Image
import copy

NoneType = type(None)

class WatermarkStableDiffusion(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 image_encoder: CLIPVisionModelWithProjection = None, # type:ignore
                 requires_safety_checker: bool = True):
                 
        super(WatermarkStableDiffusion, self).__init__(vae,
                         text_encoder,
                         tokenizer,
                         unet,
                         scheduler,
                         safety_checker,
                         feature_extractor,
                         image_encoder,
                         requires_safety_checker)
        
    @torch.no_grad()
    def __call__(self, #type:ignore
                 prompt: Union[str, List[str]] = None, # type:ignore
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 num_inference_steps: int = 50,
                 timesteps: List[int] = None, # type:ignore
                 sigmas: List[float] = None, # type:ignore
                 guidance_scale: float = 7.5,
                 negative_prompt: Union[str, List[str], NoneType] = None, # type:ignore
                 num_images_per_prompt: Optional[int] = 1,
                 eta: float = 0.0,
                 generator: Union[torch._C.Generator, List[torch._C.Generator], NoneType] = None, # type:ignore
                 latents: Optional[torch.Tensor] = None,
                 prompt_embeds: Optional[torch.Tensor] = None,
                 negative_prompt_embeds: Optional[torch.Tensor] = None,
                 ip_adapter_image: Union[Image.Image, np.ndarray, torch.Tensor, List[Image.Image], List[np.ndarray], List[torch.Tensor], NoneType] = None, # type:ignore
                 ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
                 output_type: Optional[str] = 'pil',
                 return_dict: bool = True,
                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                 guidance_rescale: float = 0.0,
                 clip_skip: Optional[int] = None,
                 callback_on_step_end: Union[Callable[[int, int, Dict], NoneType], PipelineCallback, MultiPipelineCallbacks, NoneType] = None, # type:ignore
                 callback_on_step_end_tensor_inputs: List[str] = ['latents'],
                 **kwargs):
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        callback_steps = 1
        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        device = self._execution_device
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, # type: ignore
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar: # type:ignore
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents # type: ignore
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback_on_step_end is not None and i % callback_steps == 0:
                        callback_on_step_end(step=i, timestep=t, callback_kwargs=callback_on_step_end_tensor_inputs) # type:ignore

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        image = self.decode_latents(latents)

        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
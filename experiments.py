import torch
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from PIL import Image

from watermark_stable_diffusion import WatermarkStableDiffusion
from utils import gen_callback_watermark, transform_image, save_numpy_to_image, calc_watermark_dist

def variable_latent_exp():
    seed = 229

    print("Loading Stable Diffusion 1.5")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = WatermarkStableDiffusion.from_pretrained(model_id) 
    pipe = pipe.to("cuda")

    prompt = "Claude Shannon holding a red balloon"
    print(f"Using prompt: '{prompt}'")
    eps=1e-9
    num_inference_steps = 100
    injection_stride = 20
    assert num_inference_steps % injection_stride == 0

    print("Generating clean sample image")

    exp_image_gens = []
    exp_noise_latents = []
    exp_rev_noise_latents = []

    torch.manual_seed(seed)
    image_clean = pipe(prompt,
                output_type='pil', 
                num_inference_steps=num_inference_steps).images[0] # type:ignore
    exp_image_gens.append(image_clean)
    # Reverse the noise conditioned on an EMPTY prompt
    print("Reversing image to predicted noise")
    image_tensor = transform_image(image_clean).unsqueeze(0).cuda()
    image_gen_latents = pipe.get_image_latents(image=image_tensor, sample=False)
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    image_noise = pipe(latents=image_gen_latents,
                prompt_embeds=text_embeddings,
                output_type='pil',
                guidance_scale=1,
                num_inference_steps=num_inference_steps,
                forward_process=True).images[0]
    noise_latents = pipe.get_last_latent()
    fft_latents = torch.fft.fftshift(torch.fft.fft2(noise_latents))
    fft_numpy = fft_latents.detach().cpu().numpy()
    data = np.log(abs(fft_numpy[0,0]))
    exp_rev_noise_latents.append(data)

    dist = calc_watermark_dist(fft_numpy, eps)
    print(f"Distance: {dist}")
    distances = [dist]
    
    for target_watermark_iter in range(0, num_inference_steps, injection_stride):
        torch.manual_seed(seed)
        print(f"[Target Iteration: {target_watermark_iter}]")
        watermarked_noise_callback = []
        image = pipe(prompt,
                    output_type='pil', 
                    num_inference_steps=num_inference_steps,
                    callback_on_step_end=gen_callback_watermark(target_watermark_iter, eps=eps, watermarked_noise_callback=watermarked_noise_callback),
                    callback_on_step_end_tensor_inputs=['latents']).images[0]
        img_array = np.array(image)
        exp_image_gens.append(img_array)
        exp_noise_latents.append(watermarked_noise_callback[-1])

        # Reverse the noise conditioned on an EMPTY prompt
        print("Reversing image to predicted noise")
        image_tensor = transform_image(image).unsqueeze(0).cuda()
        image_gen_latents = pipe.get_image_latents(image=image_tensor, sample=False)
        tester_prompt = ''
        text_embeddings = pipe.get_text_embedding(tester_prompt)
        image_noise = pipe(latents=image_gen_latents,
                    prompt_embeds=text_embeddings,
                    output_type='pil',
                    guidance_scale=1,
                    num_inference_steps=num_inference_steps,
                    forward_process=True,
                    stop_iter=num_inference_steps-target_watermark_iter).images[0] # Reverse up to the point when the watermark was injected
        noise_latents = pipe.get_last_latent()
        fft_latents = torch.fft.fftshift(torch.fft.fft2(noise_latents))
        fft_numpy = fft_latents.detach().cpu().numpy()

        dist = calc_watermark_dist(fft_numpy, eps)
        print(f"Distance: {dist}")
        distances.append(dist)

        data = np.log(np.abs(fft_numpy[0,0]))
        exp_rev_noise_latents.append(data)

    fig, ax = plt.subplots(2,len(exp_image_gens), figsize=(5 * len(exp_image_gens), 10))

    all_latents = exp_rev_noise_latents
    vmin = min(img.min() for img in all_latents)
    vmax = max(img.max() for img in all_latents)

    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

    for i in range(len(exp_image_gens)):
        ax[0, i].imshow(exp_image_gens[i])
        ax[1, i].imshow(exp_rev_noise_latents[i], norm=norm, cmap='viridis')

        if i > 0: ax[0, i].set_title(f"Injected Latent: {(i - 1) * injection_stride}")
        ax[1, i].set_title(f"Distance: {distances[i]}")

    ax[0, 0].set_title("Clean Latents")
    ax[0, 0].set_ylabel("Image Sample")
    ax[1, 0].set_ylabel("Reversed Noise")

    plt.savefig("results/variable_latents.png")

def initial_latent_exp():
    seed = 229
    torch.manual_seed(seed)
    eps=1e-9

    print("Loading Stable Diffusion 1.5")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = WatermarkStableDiffusion.from_pretrained(model_id) 
    pipe = pipe.to("cuda")

    prompt = "Claude Shannon holding a red balloon"
    print(f"Generating watermarked sample image: '{prompt}'")
    target_watermark_iter = 0
    num_inference_steps = 20
    init_noise_callback = []
    watermarked_noise_callback = []
    image = pipe(prompt,
                output_type='pil', 
                num_inference_steps=num_inference_steps,
                callback_on_step_end=gen_callback_watermark(target_watermark_iter, eps=eps, init_noise_callback=init_noise_callback, watermarked_noise_callback=watermarked_noise_callback),
                callback_on_step_end_tensor_inputs=['latents']).images[0]
    img_array = np.array(image)

    save_numpy_to_image(init_noise_callback[-1], "results/initial_noise.png")
    save_numpy_to_image(watermarked_noise_callback[-1], "results/watermarked_noise.png")

    # Reverse the noise conditioned on an EMPTY prompt
    print("Reversing image to predicted noise")
    image_tensor = transform_image(image).unsqueeze(0).cuda()
    image_gen_latents = pipe.get_image_latents(image=image_tensor, sample=False)
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    image_noise = pipe(latents=image_gen_latents,
                prompt_embeds=text_embeddings,
                output_type='pil', 
                guidance_scale=1,
                num_inference_steps=50,
                forward_process=True).images[0]
    noise_latents = pipe.get_last_latent()
    fft_latents = torch.fft.fftshift(torch.fft.fft2(noise_latents))
    fft_numpy = fft_latents.detach().cpu().numpy()
    data = np.log(abs(fft_numpy[0,0]))
    save_numpy_to_image(data, "results/reverse_watermarked_noise.png")

    print("Regenerating image with reversed noise")
    noise_latents = pipe.get_last_latent()
    image_noise_rev = pipe(prompt,
                latents=noise_latents,
                output_type='pil', 
                guidance_scale=1,
                num_inference_steps=20).images[0]

    print("Generating clean sample image")
    torch.manual_seed(seed)
    image_clean = pipe(prompt,
                output_type='pil', 
                num_inference_steps=20).images[0] # type:ignore

    print("Saving final plot")
    initial_noise = np.asarray(Image.open("results/initial_noise.png"))
    watermarked_noise = np.asarray(Image.open("results/watermarked_noise.png"))
    reverse_watermarked_noise = np.asarray(Image.open("results/reverse_watermarked_noise.png"))

    fig, ax = plt.subplots(2,3,figsize=(15, 10))

    ax[0, 0].imshow(image_clean)
    ax[0, 0].set_title("Initial Noise")
    ax[0, 1].imshow(img_array)
    ax[0, 1].set_title("Watermarked Noise")
    ax[0, 2].imshow(image_noise_rev)
    ax[0, 2].set_title("Inverse Noise Prediction")

    ax[1, 0].imshow(initial_noise)
    ax[1, 1].imshow(watermarked_noise)
    ax[1, 2].imshow(reverse_watermarked_noise)

    plt.savefig("results/treerings.png")
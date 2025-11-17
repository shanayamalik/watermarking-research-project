from typing import Dict
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from watermark_stable_diffusion import WatermarkStableDiffusion
import numpy as np

def save_numpy_to_image(data, name):
    dpi = 100
    height, width = data.shape
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.]) # type: ignore
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap='viridis', aspect='auto')
    plt.savefig(
        name,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0,
        transparent=True
    )
    plt.close(fig)

def generate_watermark_mask(latents: torch.Tensor, p = 0.7, w = 2) -> torch.Tensor:
    watermark_mask = torch.zeros_like(latents, dtype=torch.long)
    n_rows, n_cols = latents.shape[2:4]
    row_indices_c2 = (torch.arange(0, n_rows) - n_rows / 2) ** 2
    col_indices_c2 = (torch.arange(0, n_cols) - n_cols / 2) ** 2
    dist_grid = row_indices_c2.view(-1, 1) + col_indices_c2.view(1, -1) # Shape: (n_rows, n_cols)
    r = p * n_rows / 2
    watermark_mask = (dist_grid >= r ** 2) * (dist_grid <= (r + w) ** 2)
    watermark_mask = watermark_mask.repeat(latents.shape[0], latents.shape[1], 1, 1)
    return watermark_mask

# Callback is called right before each denoising step
def gen_callback_watermark(target_watermark_iter, eps=1e-9, init_noise_callback=None, watermarked_noise_callback=None):
    def callback_watermark(pipe: WatermarkStableDiffusion, iter: int, t: int, tensor_inputs: Dict) -> Dict:
        if iter == target_watermark_iter:
            latents = tensor_inputs['latents']
            watermark_mask = generate_watermark_mask(latents)
            fft_latents = torch.fft.fftshift(torch.fft.fft2(latents))
            fft_latents[:,0][watermark_mask[:,0]] = eps # Embed only in first latent
            
            # Save noise visual
            if init_noise_callback is not None:
                init_noise_callback.append(latents[0,0].detach().cpu().numpy())
            if watermarked_noise_callback is not None: 
                fft_numpy = fft_latents.detach().cpu().numpy()
                data = np.log(abs(fft_numpy[0,0]))
                watermarked_noise_callback.append(data)

            # Inverse fourier tranform of watermarked latents
            latents = torch.fft.ifft2(torch.fft.ifftshift(fft_latents)).type(latents.dtype)

            print(f"Embedded watermark in latent {iter}")
            return {"latents" : latents}
        return {}
    return callback_watermark

def calc_watermark_dist(latent, eps):
    target_latent = latent[0, 0]
    mask = generate_watermark_mask(torch.tensor(latent))[0, 0].numpy() # I know it is bad to convert back and forth between numpy and torch but I'm sorry, it just wasn't worth the fuss
    distances = np.zeros_like(target_latent)
    distances[mask] = np.abs(target_latent - eps)[mask]
    num_masked = np.sum(mask)
    dist = np.sum(distances).astype(float) / num_masked
    dist = np.round(dist, 3)
    print(f"Distance: {dist}")
    return dist

def transform_image(image, target_size=512):
    """resize and output -1..1"""
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0
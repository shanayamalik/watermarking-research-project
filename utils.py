from cleanfid import fid
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
import numpy as np
from PIL import Image
import os

def run_metric(metric, config):
    """Run a specific metric and return the score."""
    
    # Image quality metrics
    if metric == "fid":
        return fid.compute_fid(config["dir1"], config["dir2"])
    
    elif metric == "clip":
        clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
        
        # Load images from dir1
        image_dir = config["dir1"]
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        images = []
        for img_file in image_files:
            img = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
            images.append(torch.from_numpy(np.array(img)))
        
        images = torch.stack(images).permute(0, 3, 1, 2)
        
        # Load prompts from prompts file if provided
        prompts_file = config.get("prompts_file")
        if prompts_file and os.path.exists(prompts_file):
            import json
            with open(prompts_file) as f:
                prompts_dict = json.load(f)
            prompts = [prompts_dict.get(img_file, "") for img_file in image_files]
        else:
            prompts = config.get("prompts", [""] * len(images))
        breakpoint()
        score = clip_score_fn(images, prompts).detach()
        return round(float(score), 4)
    
    elif metric == "IS":
        # TODO: Implement Inception Score
        return 0.0
    
    # Detectability metrics
    elif metric == "binary_classifier":
        # TODO: Implement binary classifier
        return 0.0
    
    # Perceptual variability metrics
    elif metric == "lpips":
        # TODO: Implement LPIPS
        return 0.0
    
    # Robustness tests
    elif metric == "cropping":
        # TODO: Implement cropping robustness
        return 0.0
    
    elif metric == "rescaling":
        # TODO: Implement rescaling robustness
        return 0.0
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
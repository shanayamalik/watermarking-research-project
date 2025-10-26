import torch
from diffusers import StableDiffusionPipeline
import json
import argparse
import os

def main(args):
    # Load captions
    with open(args.caption_file) as f:
        data = json.load(f)
        captions = data['annotations']

    # Load Stable Diffusion 2.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    ).to(device)
    
    # Generate images
    os.makedirs(args.output_dir, exist_ok=True)
    
    prompts_dict = {}
    for i in range(args.start, min(args.end, len(captions))):
        caption_data = captions[i]
        caption = caption_data['caption']
        image_id = caption_data['image_id']
        image = pipe(caption, num_inference_steps=50).images[0]
        image.save(f"{args.output_dir}/{image_id:012d}.png")
        prompts_dict[f"{image_id:012d}.png"] = caption
        print(f"Generated {image_id:012d}: {caption}")
    
    # Save prompts to file
    prompts_file = os.path.join(args.output_dir, "prompts.json")
    with open(prompts_file, 'w') as f:
        json.dump(prompts_dict, f, indent=2)
    print(f"Saved prompts to {prompts_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_file", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="outputs/sd2.1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    args = parser.parse_args()
    main(args)


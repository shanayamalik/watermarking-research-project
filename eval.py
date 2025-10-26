from cleanfid import fid
import argparse
import os
import json
import yaml
from utils import run_metric

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    base_config = config["base_config"]
    results = {}
    
    # Run each evaluation mode
    for mode in config["evaluation_modes"]:
        for metric in mode:
            print(f"Running {mode}/{metric}...")
            score = run_metric(metric, base_config)
            results[mode][metric] = score
            print(f"  {metric}: {score}")
    
    # Save results
    output_file = base_config["output_file"]
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    args = parser.parse_args()
    main(args.config)
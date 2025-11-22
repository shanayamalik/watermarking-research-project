from cleanfid import fid
import argparse
import os
import json

def main(args):
    results = {}
    if "fid" in args.mode:
        score = fid.compute_fid(args.dir1, args.dir2)
        print(score)
    else:
        raise ValueError(f"Mode {args.mode} not found")
    results["fid"] = score
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, nargs='+', required=True, choices=["fid"])
    parser.add_argument("--dir1", type=str, required=True)
    parser.add_argument("--dir2", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
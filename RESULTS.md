# Watermarking Experiment Results

## What was implemented
- Added and exercised a blockwise DCT watermarking module alongside existing LSB and Fourier implementations, with embedded self-tests and config hooks for evaluation.
- Built a reusable attack runner that orchestrates compression, rescaling, cropping, Gaussian noise, and Gaussian blur attacks while logging watermark confidence, PSNR, and SSIM per attack.
- Updated `shanaya/README.md` to document setup, evaluation entry points, and per-method self-tests; expanded `eval_config.yaml` and utility wiring to support configurable suites; and included the new `attack_runner.py` and `dct_watermarking.py` modules.
- Ported the Stable Diffusion latent ring-mask watermark (from `shai/` and `alex/`) into a new `StableDiffusionWatermarking` class, wired it into the attack runner/configs, and evaluated it against the same perturbations.

## Attack experiment summary
The following suites were run with the sample mandrill image (auto-generated watermarks where needed). Confidence is reported on a 0–1 scale, alongside PSNR/SSIM versus the pristine watermarked image.

### LSB (compression only suite)
- JPEG q80: confidence 1.00, PSNR 361.20, SSIM 1.00
- JPEG q30: confidence 1.00, PSNR 361.20, SSIM 1.00

### Fourier suite
- JPEG q80/q30: confidence 1.00 (all), PSNR ~361.20, SSIM 1.00
- Rescale 75%: confidence 1.00, PSNR 25.28, SSIM 0.84
- Center crop 85%: confidence 1.00, PSNR 13.17, SSIM 0.12
- Gaussian noise σ=5: confidence 1.00, PSNR 34.10, SSIM 0.95
- Gaussian blur k=5: confidence 1.00, PSNR 22.81, SSIM 0.69

### DCT suite
- JPEG q80/q30: confidence 1.00 (all), PSNR ~361.20, SSIM 1.00
- Rescale 75%: confidence 0.00, PSNR 25.28, SSIM 0.84
- Center crop 85%: confidence 0.00, PSNR 13.17, SSIM 0.12
- Gaussian noise σ=5: confidence 0.83, PSNR 34.10, SSIM 0.95
- Gaussian blur k=5: confidence 0.09, PSNR 22.81, SSIM 0.69

### Stable Diffusion-inspired suite
- JPEG q80/q30: confidence 0.99 (all), PSNR ~361.20, SSIM 1.00
- Rescale 75%: confidence 0.00, PSNR 34.05, SSIM 0.96
- Center crop 85%: confidence 0.12, PSNR 10.71, SSIM 0.41
- Gaussian noise σ=5: confidence 0.49, PSNR 34.23, SSIM 0.82
- Gaussian blur k=5: confidence 0.00, PSNR 31.21, SSIM 0.93

## Visualization suggestions
Use the JSON outputs under `shanaya/results/attacks` as inputs to plot confidence and quality metrics.

### Bar chart of confidence by attack
```python
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

results_dir = Path("shanaya/results/attacks")
method_files = {
    "LSB": results_dir / "lsb_results.json",
    "Fourier": results_dir / "fourier_results.json",
    "DCT": results_dir / "dct_results.json",
    "Stable Diffusion": results_dir / "stable_diffusion_results.json",
}

results = {}
all_attacks = set()
for method, path in method_files.items():
    with open(path) as f:
        data = json.load(f)
    results[method] = data
    for item in data:
        all_attacks.add(item["name"])
all_attacks = sorted(all_attacks)

fig, ax = plt.subplots(figsize=(12, 4))
width = 0.18
x = np.arange(len(all_attacks))
for idx, (method, data) in enumerate(results.items()):
    attack_map = {item["name"]: item["confidence"] for item in data}
    confidences = [attack_map.get(name, np.nan) for name in all_attacks]
    ax.bar(x + idx * width, confidences, width=width, label=method)

ax.set_xticks(x + width * (len(results) - 1) / 2)
ax.set_xticklabels(all_attacks, rotation=30, ha="right")
ax.set_ylabel("Confidence")
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Watermark confidence across attacks")
plt.tight_layout()
plt.savefig("confidence_by_attack.png", dpi=150)
print("Saved confidence_by_attack.png")
```

### PSNR/SSIM line plots for each method
```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

for method in ["lsb", "fourier", "dct", "stable_diffusion"]:
    path = Path(f"shanaya/results/attacks/{method}_results.json")
    with open(path) as f:
        data = json.load(f)

    attacks = [item["name"] for item in data]
    psnr = [item["psnr"] for item in data]
    ssim = [item["ssim"] for item in data]

    fig, ax1 = plt.subplots(figsize=(8, 3))
    color = "tab:blue"
    ax1.set_xlabel("Attack")
    ax1.set_ylabel("PSNR", color=color)
    ax1.plot(attacks, psnr, marker="o", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticklabels(attacks, rotation=30, ha="right")

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("SSIM", color=color)
    ax2.plot(attacks, ssim, marker="s", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1.05)

    plt.title(f"Image fidelity after attacks ({method.upper().replace('_', ' ')})")
    fig.tight_layout()
    plt.savefig(f"{method}_fidelity.png", dpi=150)
    print(f"Saved {method}_fidelity.png")
```

### How the plots are stored in this repository
To avoid committing binary images, the plots generated above were saved to the repository root and then converted to Base64-encoded text files:

- `confidence_by_attack.png.base64`
- `lsb_fidelity.png.base64`
- `fourier_fidelity.png.base64`
- `dct_fidelity.png.base64`
- `stable_diffusion_fidelity.png.base64`

To reconstruct the original PNGs locally, run from the repository root:

```bash
for f in *.png.base64; do
    base64 --decode "$f" > "${f%.base64}"
done
```
This will write the `.png` files alongside their `.png.base64` sources while leaving the Base64 text files untouched.

## Summary of code changes in `shanaya/`
- **README.md:** Documented the four supported methods (adding the Stable Diffusion-inspired ring mask), setup, config-driven evaluation, direct attack suites, and per-method self-tests.
- **configs/eval_config.yaml:** Added attack suite definitions for LSB, Fourier, DCT, and Stable Diffusion-inspired methods, including compression, rescaling, cropping, noise, and blur parameters, plus evaluation mode listings.
- **eval.py and requirements.txt:** Light touch-ups to keep evaluation entry point and dependencies aligned with the new tooling.
- **src/attack_runner.py (new):** Orchestrates configurable attack suites, ensures watermarked inputs are prepared, applies attacks, computes confidence/PSNR/SSIM, and writes structured JSON outputs.
- **src/dct_watermarking.py (new):** Implements blockwise DCT watermark embedding/extraction with configurable coefficient pairs, strength parameter, delimiter-based payload recovery, and a CLI harness.
- **src/fourier_watermarking.py & src/lsb_watermarking.py:** Minor updates to support confidence scoring and CLI flows used by the attack runner.
- **src/stable_diffusion_watermarking.py (new):** Ports the `shai/` ring-mask latent watermark into an image-space routine that boosts a circular Fourier band on a Stable Diffusion 2.1 sample, plus confidence scoring.
- **utils.py:** Expanded helpers for watermarking and attack pipelines to align with the new modules and configs.

## Step-by-step attack evaluation
1. Install dependencies (from the repo root):
   ```bash
   pip install -r shanaya/requirements.txt
   ```
2. Ensure the Stable Diffusion cover image from `alex/outputs/sd2.1/000000006818.png` is present (regenerate with `alex/generate_sd.py` if needed).
3. Run any configured attack suite, for example the Stable Diffusion-inspired ring mask:
   ```bash
   python shanaya/src/attack_runner.py --config shanaya/configs/eval_config.yaml --suite stable_diffusion_attack_suite
   ```
   Substitute `lsb_attack_suite`, `fourier_attack_suite`, or `dct_attack_suite` to evaluate the other methods.
4. Regenerate plots (optional) after attacks complete:
   ```bash
   python - <<'PY'
   # plotting code from the Visualization section
   PY
   ```
5. Convert the generated `.png` plots to `.png.base64` text files for sharing (if rerun):
   ```bash
   for f in *.png; do base64 "$f" > "$f.base64"; done
   ```

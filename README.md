# Classic Watermarking Experiments

This folder implements spatial- and frequency-domain watermarking baselines and
configurable robustness experiments. Four watermarking methods are supported:

- **LSB (spatial)** – embeds bits in pixel least-significant bits.
- **Fourier (frequency)** – inserts a subtle pattern into the green channel FFT.
- **DCT (frequency)** – encodes bits via pairwise relationships between
  mid-frequency DCT coefficients in the luminance channel.
- **Stable Diffusion (latent-inspired)** – ports the ring-mask latent
  watermarking idea from `shai/` into image space by boosting a circular band in
  the Fourier magnitude of a Stable Diffusion 2.1 sample (from `alex/outputs`).

## Setup

```bash
pip install -r requirements.txt
```

The default configs expect the sample mandrill image at `images/4.2.03.tiff`
(included in the repo). Outputs are written to `results/`.

## Quick starts

### Evaluate configured metrics

Run the pre-defined robustness metrics (JPEG compression for each method and
full attack suites) using the shared YAML config:

```bash
python eval.py --config configs/eval_config.yaml
```

Results are saved to `results/evaluation_results.json`.

The Stable Diffusion suite uses `alex/outputs/sd2.1/000000006818.png` as its
default cover; swap this path in `configs/eval_config.yaml` if you regenerate
latent samples.

### Run an attack suite directly

Attack suites live under `base_config.attack_suites` in the config. Each suite
defines the watermark method, image paths, optional message, and a list of
attacks. To run one suite end-to-end and write per-attack metrics:

```bash
python src/attack_runner.py --config configs/eval_config.yaml --suite lsb_attack_suite
```

This produces attacked images under `results/attacks/<suite>` and a JSON file of
confidence/PSNR/SSIM scores.

To exercise the Stable Diffusion-inspired watermark attacks specifically:

```bash
python src/attack_runner.py --config configs/eval_config.yaml --suite stable_diffusion_attack_suite
```

### Method self-tests

Each watermark implementation includes a simple CLI harness that embeds a
default message into `images/4.2.03.tiff`, saves a watermarked image, and prints
the extracted message and confidence:

```bash
python src/lsb_watermarking.py
python src/fourier_watermarking.py
python src/dct_watermarking.py
```

These scripts also generate example artifacts under `results/` if they do not
already exist.
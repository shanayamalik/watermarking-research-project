# Watermarking Research Project

This repository explores spatial- and frequency-domain image watermarking techniques, combining classic methods with Stable Diffusion-based latent experiments. The current documentation highlights the experimental plan, recorded results, and a catalog of the original files for quick orientation. The latest iteration ports the ring-mask latent watermark from `shai/`/`alex/` into the `shanaya/` attack runner so it can be evaluated alongside the classic image-space methods.

## Quick links
- [Implementation plan](PLAN.md)
- [Experiment results and detailed metrics](RESULTS.md)
- [Initial file inventory](INITIAL-FILES.md)

## Experiment outcomes
Watermark confidence and fidelity were evaluated across LSB (least significant bit), Fourier, and DCT (discrete cosine transform) methods using compression, resizing, cropping, noise, and blur attacks. Summary plots are included below; see [RESULTS.md](RESULTS.md) for the underlying tables and reproduction instructions.

- ![Bar chart showing watermark confidence across attacks for LSB, Fourier, and DCT methods](confidence_by_attack.png) — The horizontal axis lists each attack (JPEG quality levels, rescaling, cropping, Gaussian noise, and Gaussian blur), and the vertical axis reports watermark confidence on a 0–1 unitless scale.
- ![Line plot of PSNR/SSIM after attacks for LSB watermarking](lsb_fidelity.png) — The horizontal axis enumerates the same attacks, while the left vertical axis shows PSNR (Peak Signal-to-Noise Ratio, in decibels) and the right vertical axis shows SSIM (Structural Similarity Index Measure, unitless on a 0–1 scale).
- ![Line plot of PSNR/SSIM after attacks for Fourier watermarking](fourier_fidelity.png) — The horizontal axis lists the attacks; PSNR (Peak Signal-to-Noise Ratio, decibels) is plotted on the left vertical axis and SSIM (Structural Similarity Index Measure, 0–1 unitless) on the right vertical axis.
- ![Line plot of PSNR/SSIM after attacks for DCT watermarking](dct_fidelity.png) — The horizontal axis lists the attacks; PSNR (Peak Signal-to-Noise Ratio, decibels) is plotted on the left vertical axis and SSIM (Structural Similarity Index Measure, 0–1 unitless) on the right vertical axis.
- ![Line plot of PSNR/SSIM after attacks for the Stable Diffusion-inspired watermark](stable_diffusion_fidelity.png) — Attacks align with the other full suites; PSNR (dB) and SSIM (0–1) are reported on dual axes.

Files added or modified while executing the plan include the expanded `shanaya/README.md`, configuration (`shanaya/configs/eval_config.yaml`), evaluation entry point (`shanaya/eval.py`), dependency lock (`shanaya/requirements.txt`), attack outputs (`shanaya/results/attacks/*.json`), a reusable attack runner (`shanaya/src/attack_runner.py`), the new DCT watermarking module (`shanaya/src/dct_watermarking.py`), updates to Fourier and LSB implementations (`shanaya/src/fourier_watermarking.py`, `shanaya/src/lsb_watermarking.py`), and shared utilities (`shanaya/utils.py`).

Key observations from the current suites:
- LSB and Fourier watermarks maintained perfect confidence under JPEG compression, while Fourier also survived resizing, cropping, noise, and blur with high scores.
- DCT watermarks stayed resilient to compression and modest noise but were fragile to resizing, cropping, and blur, indicating room for parameter tuning or detector improvements.
- The Stable Diffusion-inspired watermark resists JPEG compression and moderate noise but loses confidence under blur and rescaling, showing that the latent-style ring mask is sensitive to spatial perturbations.

## Summary and next steps
The project now includes a blockwise DCT watermarking module alongside existing LSB and Fourier implementations, a Stable Diffusion-inspired ring watermark in image space, a reusable attack runner for common perturbations, and refreshed documentation/configuration to support the evaluation flow. Future iterations can extend the success criteria, integrate the attack framework with full latent diffusion decoding, and add broader perceptual metrics to deepen the analysis.

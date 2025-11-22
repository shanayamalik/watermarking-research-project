# Watermarking Research Project: Implementation Plan (Status Update)

## Context and goals
- The repository already contains classic spatial- and frequency-domain watermarking baselines (LSB, Fourier) plus Stable Diffusion latent watermark experiments. We need to extend these with a DCT-style watermark and automated robustness attacks to evaluate detectability, imperceptibility, and survivability across techniques.
- Primary codebases: `shai/` (Stable Diffusion latent watermarking experiments) and `shanaya/` (classical image-domain watermarking implementations and compression tests). The `alex/` utilities provide an evaluation scaffold for dataset-level metrics (e.g., FID) if needed for broader benchmarking.

## Current state
- **LSB watermarking (shanaya/src/lsb_watermarking.py):** Complete spatial-domain embed/extract with confidence scoring and a CLI test harness using the mandrill image; also used by the existing JPEG compression robustness helper.
- **Fourier watermarking (shanaya/src/fourier_watermarking.py):** Embeds a subtle mark in the FFT of the green channel and scores detection; integrated into a compression test similar to the LSB flow.
- **Stable Diffusion latent watermark experiments (shai/experiments.py):** Two scripts inject/recover Fourier-style watermarks in SD 1.5 latents, visualize reversed noise, and save comparison plots (variable injection step sweep + initial latent study). The latent ring-mask was also ported into `shanaya/src/stable_diffusion_watermarking.py` so it can be attacked alongside the classic image-domain methods using Alex's SD 2.1 samples.
- **Project layout reference:** README summarizes the roles of `alex/`, `shai/`, and `shanaya/` directories and their current utilities and configs.

## Work plan and status
1. **Define the research question and success criteria in writing**
   - *Status: Partially completed.* Hypothesis and metrics captured in earlier docs; still need a concise written hypothesis and success criteria aligned to the new attack results (e.g., survival thresholds by attack type) for the final report.

2. **Implement DCT-based watermarking module in `shanaya/src/`**
   - *Status: Completed.* `dct_watermarking.py` implements blockwise embedding/extraction with configurable coefficient pairs, delimiter-based payload handling, confidence scoring, and a CLI self-test that writes artifacts to `results/`.

3. **Automated attack/evaluation script (shanaya focus, reusable for shai)**
   - *Status: Completed for image-domain methods.* `attack_runner.py` reads YAML suites, ensures watermarked inputs exist, applies compression/rescaling/cropping/noise/blur attacks, and logs confidence/PSNR/SSIM JSON results per suite. Hooks for SD latent outputs remain future work.

4. **Evaluation integration and metrics expansion**
   - *Status: Completed for classic attacks plus a Stable Diffusion-inspired ring watermark.* `configs/eval_config.yaml` now lists suites for LSB, Fourier, DCT, and the SD ring watermark covering compression, resizing, cropping, noise, and blur. Additional metrics (e.g., FID/LPIPS) and deeper latent adapters are still pending.

5. **Documentation and experiment scripts**
   - *Status: Completed for `shanaya/`.** README and configs explain setup, CLI self-tests, and running suites. Added Stable Diffusion-inspired attack instructions; a brief README refresh in `shai/` plus reproduction notes for end-to-end latent decoding remain future work.

6. **Future considerations**
   - *Status: Not started.* Temporal/video extensions, learned detectors, and dataset-scale perceptual metrics remain future enhancements once core image benchmarks are finalized.

## Deliverables checklist
- [x] `dct_watermarking.py` + tests/artifacts; config hooks for running it.
- [x] Automated attack runner with configurable attack suites and logged results.
- [x] Updated evaluation utilities/configs to cover new attacks and watermark methods.
- [x] Documentation (PLAN.md, README updates, run instructions) and curated experiment outputs supporting the research conclusions.
- [ ] Expanded success criteria write-up and deeper SD latent integration (beyond the current ring-mask image-space port) into the attack/eval pipeline.
- [ ] Optional advanced metrics (FID/LPIPS), video/temporal experiments, and learned detector baselines.

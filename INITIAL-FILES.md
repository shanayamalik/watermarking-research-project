# Initial File Inventory

This document summarizes the original layout and contents of the repository prior to this documentation refresh.

## Top-level files
- `report-template.tex`: IEEE conference paper template pre-filled with sample author blocks, abstract text, formatting guidance, and commented examples for figures and tables.

## alex
Utility scripts for generating Stable Diffusion 2.1 samples and evaluating them with metrics defined via YAML configuration.

- `README.md`: Quickstart steps for installing `clean-fid`, downloading the COCO2017 subset, and running the generation and evaluation scripts.
- `configs/eval_config.yaml`: Configuration defining source directories, prompt metadata path, output JSON file, and grouped evaluation modes (image quality, detectability, variability, robustness).
- `generate_sd.py`: Loads captions from a metadata JSON file, runs the Stable Diffusion 2.1 pipeline on GPU if available, saves numbered PNGs to `outputs/sd2.1/`, and records prompt text to `prompts.json`.
- `eval.py`: Reads the YAML config, iterates through configured metrics, delegates to `utils.run_metric`, and writes aggregated results to the configured output JSON path.
- `utils.py`: Implements metric helpers including FID calculation, CLIP score placeholder with a breakpoint for debugging, and stubs for additional metrics such as Inception Score, binary classifiers, LPIPS, and robustness checks.
- `outputs/sd2.1/`: Sample generated images (`*.png`) and the `prompts.json` file mapping image filenames to captions.
- `scores/initial_scores1.json`: Example metric output destination referenced by the config.
- `.gitignore`: Ignores virtual environment artifacts and large data directories (contents not shown here).

## evals
Minimal baseline scripts mirroring the `alex` pipeline but focused only on FID computation.

- `README.md`: Download and execution instructions for generating images and evaluating FID.
- `generate_sd.py`: Generates Stable Diffusion 2.1 images from COCO-style caption metadata and stores them under `outputs/sd2.1/`.
- `eval.py`: Computes FID between two directories, prints the score, and writes it to a JSON file.
- `outputs/sd2.1/`: Sample generated PNG images reused for quick evaluation.

## shai
Experiments that inject and recover frequency-domain watermarks in Stable Diffusion 1.5 latents, including visualization utilities.

- `README.md`: Placeholder file (currently empty).
- `main.py`: Entry point that currently runs `variable_latent_exp` from `experiments.py`.
- `experiments.py`: Defines two experiments—`variable_latent_exp` sweeps watermark injection across diffusion steps while measuring watermark distance and visualizing reversed noise, and `initial_latent_exp` embeds a watermark at the first step then reverses and regenerates images for comparison.
- `watermark_stable_diffusion.py`: Custom Stable Diffusion pipeline subclass exposing latent accessors, text/image embedding helpers, and overridden denoising to support forward/backward diffusion with callback hooks for watermarking.
- `utils.py`: Helper functions for generating watermark masks in Fourier space, callback construction for injecting watermarks during diffusion, distance computation, image tensor transforms, and plotting arrays to image files.
- `generate_test.ipynb`: Notebook scaffold for interactive experimentation (contents not summarized here).
- `results/`: Stored artifacts from experiments, including intermediate noise visualizations and generated samples (e.g., `variable_latents_v1.png`, `watermarked_noise.png`, `shannon_redballoon.png`).
- `.gitignore`: Excludes large model checkpoints and cache directories for experiments.

## shanaya
Classic watermarking approaches and robustness evaluations using both LSB and Fourier-domain techniques.

- `README.md`: Project heading for the watermarking experiments.
- `requirements.txt`: Python dependencies for image processing and evaluation (e.g., OpenCV, NumPy).
- `configs/eval_config.yaml`: Defines paths for LSB and Fourier watermarked images, compression quality, and evaluation outputs; enables robustness metrics for compression tests.
- `eval.py`: Loads the YAML configuration, runs selected metrics via `utils.run_metric`, and writes results to JSON.
- `utils.py`: Metric dispatcher with implementations for LSB and Fourier compression robustness tests, plus placeholders for additional metrics.
- `src/lsb_watermarking.py`: Implements least-significant-bit watermark embedding, extraction, confidence scoring, and a CLI-style test harness for the sample mandrill image.
- `src/fourier_watermarking.py`: Embeds subtle watermarks in the Fourier domain (green channel) and estimates detection confidence; includes a self-test using the mandrill image.
- `src/safe_compression_test.py`: Safety-wrapped compression test for LSB watermarks that limits pixel scanning during extraction to avoid hangs.
- `images/4.2.03.tiff`: Mandrill test image used by the watermarking scripts.
- `results/`: Example outputs, including watermarked images, a temporary compressed file, and `evaluation_results.json` capturing compression metric scores.

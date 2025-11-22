import json
import os
import sys
from typing import Any, Dict

import cv2


ROOT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from lsb_watermarking import LSBWatermarking  # noqa: E402
from fourier_watermarking import FourierWatermarking  # noqa: E402
from dct_watermarking import DCTWatermarking  # noqa: E402
from stable_diffusion_watermarking import StableDiffusionWatermarking  # noqa: E402


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _prepare_lsb_image(config: Dict[str, Any]) -> str:
    lsb_cfg = config.get("lsb", {})
    cover_image = _resolve_path(lsb_cfg.get("cover_image", "images/4.2.03.tiff"))
    watermarked_image = _resolve_path(lsb_cfg.get("watermarked_image", "results/mandrill_watermarked_lsb.png"))
    message = lsb_cfg.get(
        "message",
        "This is a secret watermark embedded using LSB technique!",
    )

    if not os.path.exists(watermarked_image):
        watermarker = LSBWatermarking()
        _ensure_dir(os.path.dirname(watermarked_image))
        watermarker.embed_watermark(cover_image, message, watermarked_image)

    return watermarked_image


def _prepare_fourier_image(config: Dict[str, Any]) -> str:
    fourier_cfg = config.get("fourier", {})
    cover_image = _resolve_path(fourier_cfg.get("cover_image", "images/4.2.03.tiff"))
    watermarked_image = _resolve_path(
        fourier_cfg.get("watermarked_image", "results/mandrill_watermarked_fourier.png")
    )

    if not os.path.exists(watermarked_image):
        watermarker = FourierWatermarking()
        _ensure_dir(os.path.dirname(watermarked_image))
        watermarker.embed_watermark(cover_image, watermarked_image)

    return watermarked_image


def _prepare_dct_image(config: Dict[str, Any]) -> str:
    dct_cfg = config.get("dct", {})
    cover_image = _resolve_path(dct_cfg.get("cover_image", "images/4.2.03.tiff"))
    watermarked_image = _resolve_path(
        dct_cfg.get("watermarked_image", "results/mandrill_watermarked_dct.png")
    )
    message = dct_cfg.get("message", "This is a discrete cosine transform watermark!")

    if not os.path.exists(watermarked_image):
        watermarker = DCTWatermarking()
        _ensure_dir(os.path.dirname(watermarked_image))
        watermarker.embed_watermark(cover_image, message, watermarked_image)

    return watermarked_image


def _prepare_stable_diffusion_image(config: Dict[str, Any]) -> str:
    sd_cfg = config.get("stable_diffusion", {})
    cover_image = _resolve_path(
        sd_cfg.get(
            "cover_image",
            os.path.join("..", "alex", "outputs", "sd2.1", "000000006818.png"),
        )
    )
    watermarked_image = _resolve_path(
        sd_cfg.get("watermarked_image", "results/stable_diffusion_watermarked.png")
    )

    if not os.path.exists(watermarked_image):
        watermarker = StableDiffusionWatermarking()
        _ensure_dir(os.path.dirname(watermarked_image))
        watermarker.embed_watermark(cover_image, watermarked_image)

    return watermarked_image


def _compute_confidence_for_method(method: str, image_path: str, config: Dict[str, Any]) -> float:
    if method == "lsb":
        watermarker = LSBWatermarking()
        message = config.get("lsb", {}).get(
            "message", "This is a secret watermark embedded using LSB technique!"
        )
        return watermarker.calculate_confidence_score(image_path, message)

    if method == "fourier":
        watermarker = FourierWatermarking()
        return watermarker.calculate_confidence_score(image_path)

    if method == "dct":
        watermarker = DCTWatermarking()
        message = config.get("dct", {}).get(
            "message", "This is a discrete cosine transform watermark!"
        )
        return watermarker.calculate_confidence_score(image_path, message)

    if method in {"stable_diffusion", "stable-diffusion", "sd"}:
        watermarker = StableDiffusionWatermarking()
        return watermarker.calculate_confidence_score(image_path)

    raise ValueError(f"Unknown watermarking method: {method}")


def run_fourier_compression_test(config: Dict[str, Any]) -> float:
    """Run compression robustness test for Fourier watermarking."""

    quality = config.get("compression_quality", 50)
    watermarked_image = _prepare_fourier_image(config)

    image = cv2.imread(watermarked_image)
    compressed_path = _resolve_path("results/temp_fourier_compressed.jpg")
    cv2.imwrite(compressed_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    watermarker = FourierWatermarking()
    confidence_score = watermarker.calculate_confidence_score(compressed_path)

    if os.path.exists(compressed_path):
        os.remove(compressed_path)

    return confidence_score


def run_compression_test(config: Dict[str, Any]) -> float:
    """Run compression robustness test for LSB watermarking."""

    quality = config.get("compression_quality", 50)
    watermarked_image = _prepare_lsb_image(config)
    message = config.get("lsb", {}).get(
        "message", "This is a secret watermark embedded using LSB technique!"
    )

    image = cv2.imread(watermarked_image)
    compressed_path = _resolve_path("results/temp_compressed.jpg")
    cv2.imwrite(compressed_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    watermarker = LSBWatermarking()
    confidence = watermarker.calculate_confidence_score(compressed_path, message)

    if os.path.exists(compressed_path):
        os.remove(compressed_path)

    return confidence


def run_dct_compression_test(config: Dict[str, Any]) -> float:
    """Run compression robustness test for DCT watermarking."""

    quality = config.get("compression_quality", 50)
    watermarked_image = _prepare_dct_image(config)
    message = config.get("dct", {}).get(
        "message", "This is a discrete cosine transform watermark!"
    )

    image = cv2.imread(watermarked_image)
    compressed_path = _resolve_path("results/temp_dct_compressed.jpg")
    cv2.imwrite(compressed_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    watermarker = DCTWatermarking()
    confidence = watermarker.calculate_confidence_score(compressed_path, message)

    if os.path.exists(compressed_path):
        os.remove(compressed_path)

    return confidence


def run_attack_suite(config: Dict[str, Any], suite_name: str) -> Dict[str, Any]:
    from attack_runner import AttackRunner

    suites = config.get("attack_suites", {})
    if suite_name not in suites:
        raise ValueError(f"Unknown attack suite: {suite_name}")

    suite_config = suites[suite_name].copy()
    if "watermarked_image" in suite_config:
        suite_config["watermarked_image"] = _resolve_path(suite_config["watermarked_image"])
    if "cover_image" in suite_config and suite_config["cover_image"]:
        suite_config["cover_image"] = _resolve_path(suite_config["cover_image"])
    if "output_dir" in suite_config:
        suite_config["output_dir"] = _resolve_path(suite_config["output_dir"])
    if "results_file" in suite_config:
        suite_config["results_file"] = _resolve_path(suite_config["results_file"])

    runner = AttackRunner(suite_config)
    results = runner.run()

    avg_confidence = float(sum(r["confidence"] for r in results) / len(results)) if results else 0.0
    min_confidence = float(min((r["confidence"] for r in results), default=0.0))
    summary = {
        "suite": suite_name,
        "average_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "num_attacks": len(results),
        "attacks": results,
    }

    results_path = suite_config.get("results_file")
    if results_path:
        _ensure_dir(os.path.dirname(results_path))
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def run_metric(metric: str, base_config: Dict[str, Any], full_config: Dict[str, Any] | None = None) -> Any:
    """Run a specific metric and return the score."""

    config = full_config or {"base_config": base_config, **base_config}
    suites = config.get("base_config", {}).get("attack_suites", {}) or base_config.get("attack_suites", {})

    if metric == "compression":
        return run_compression_test(base_config)

    if metric == "fourier_compression":
        return run_fourier_compression_test(base_config)

    if metric == "dct_compression":
        return run_dct_compression_test(base_config)

    if metric == "stable_diffusion_compression":
        return run_attack_suite(
            base_config | {"attack_suites": suites}, "stable_diffusion_attack_suite"
        )

    if metric.endswith("_attack_suite"):
        suite_name = metric
        if suite_name not in suites:
            raise ValueError(f"Attack suite '{suite_name}' not found in config")
        return run_attack_suite(base_config | {"attack_suites": suites}, suite_name)

    raise ValueError(f"Unknown metric: {metric}")


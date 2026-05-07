"""
Perceptual and timing evaluation for NL2ECF-SRCNN.

This script compares the enhanced (super-resolved) images produced by the
NL2ECF-SRCNN pipeline against their corresponding low-resolution inputs and
high-resolution ground truth images. It reports:

- PSNR/SSIM/MSE vs HR for LR and SR images
- Perceived sharpness improvement (variance of Laplacian) from LR to SR
- Artifact reduction as MSE reduction percentage from LR->HR to SR->HR
- Average per-image generation time (preprocess + predict + postprocess)

Usage examples:
  python -m evaluation.perceptual_eval --max-samples 50
  python -m evaluation.perceptual_eval --model-path logs/nl2ecf_srcnn_model.h5 --save-csv logs/perceptual_eval.csv
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm  # type: ignore

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.model_prediction import (  # noqa: E402
    preprocess_image,
    model_predictions,
    nl2ecf_postprocess,
)
from model.model_architecture import NL2ECF_SRCNN_model  # noqa: E402
from model.quality_metrics import evaluate_metrics  # noqa: E402


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def compute_sharpness_variance_laplacian(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def load_pairs(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, "r") as f:
        return json.load(f)


def load_model(weights_path: str):
    model = NL2ECF_SRCNN_model()
    model.load_weights(weights_path)
    return model


def evaluate_sample(hr_path: str, lr_path: str, model) -> Dict[str, float]:
    # Load images
    hr_img = load_image(hr_path)
    lr_img = load_image(lr_path)

    # Baseline metrics (LR vs HR)
    baseline = evaluate_metrics(hr_img, lr_img)

    # Run NL2ECF pipeline and time it
    start = time.perf_counter()
    preprocessed, ycbcr = preprocess_image(lr_img)
    pred = model_predictions(model, preprocessed)
    sr_img = nl2ecf_postprocess(pred, ycbcr)
    elapsed = time.perf_counter() - start

    # Enhanced metrics (SR vs HR)
    enhanced = evaluate_metrics(hr_img, sr_img)

    # Sharpness
    sharp_lr = compute_sharpness_variance_laplacian(lr_img)
    sharp_sr = compute_sharpness_variance_laplacian(sr_img)

    # Improvements
    psnr_improve_db = enhanced["PSNR"] - baseline["PSNR"]
    ssim_improve = enhanced["SSIM"] - baseline["SSIM"]

    # MSE reduction percent; guard against zero
    mse_lr = baseline["MSE"]
    mse_sr = enhanced["MSE"]
    if mse_lr > 0:
        mse_reduction_pct = (mse_lr - mse_sr) / mse_lr * 100.0
    else:
        mse_reduction_pct = 0.0

    # Sharpness improvement percent; guard against zero
    if sharp_lr > 0:
        sharpness_improve_pct = (sharp_sr - sharp_lr) / sharp_lr * 100.0
    else:
        sharpness_improve_pct = 0.0

    return {
        # Baseline to HR
        "psnr_lr": float(baseline["PSNR"]),
        "ssim_lr": float(baseline["SSIM"]),
        "mse_lr": float(mse_lr),
        # Enhanced to HR
        "psnr_sr": float(enhanced["PSNR"]),
        "ssim_sr": float(enhanced["SSIM"]),
        "mse_sr": float(mse_sr),
        # Improvements
        "psnr_improve_db": float(psnr_improve_db),
        "ssim_improve": float(ssim_improve),
        "mse_reduction_pct": float(mse_reduction_pct),
        "sharpness_lr": float(sharp_lr),
        "sharpness_sr": float(sharp_sr),
        "sharpness_improve_pct": float(sharpness_improve_pct),
        # Timing
        "inference_seconds": float(elapsed),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for k in keys:
        out[k] = float(np.mean([r[k] for r in rows]))
    # Also include timing percentiles for better reporting
    times = np.array([r["inference_seconds"] for r in rows], dtype=float)
    out["inference_seconds_p10"] = float(np.percentile(times, 10))
    out["inference_seconds_p90"] = float(np.percentile(times, 90))
    out["num_samples"] = len(rows)
    return out


def find_paths(hr_name: str, lr_name: str, hr_base: str, lr_base: str) -> Tuple[str, str]:
    return (
        os.path.join(hr_base, f"HR_{hr_name}"),
        os.path.join(lr_base, lr_name),
    )


def run_perceptual_eval(
    json_path: str = os.path.join(PROJECT_ROOT, "data/hr_lr_pairs.json"),
    model_path: str = os.path.join(PROJECT_ROOT, "logs/nl2ecf_srcnn_model.h5"),
    hr_dir: str = os.path.join(PROJECT_ROOT, "data/high_resolution_images"),
    lr_dir: str = os.path.join(PROJECT_ROOT, "data/low_resolution_images"),
    max_samples: int | None = None,
    save_csv: str | None = None,
    print_summary: bool = True,
    return_rows: bool = False,
):
    print("Loading model...")
    model = load_model(model_path)

    print("Loading HR-LR pairs...")
    pairs = load_pairs(json_path)

    rows: List[Dict[str, float]] = []
    sample_count = 0

    # Iterate pairs; for each HR image, evaluate all its LR counterparts
    for hr_name, lr_list in tqdm(pairs.items(), desc="Evaluating"):
        for lr_name in lr_list:
            if max_samples is not None and sample_count >= max_samples:
                break
            hr_path, lr_path = find_paths(hr_name, lr_name, hr_dir, lr_dir)
            if not os.path.exists(hr_path) or not os.path.exists(lr_path):
                continue
            try:
                row = evaluate_sample(hr_path, lr_path, model)
                rows.append(row)
                sample_count += 1
            except Exception as e:
                print(f"Skipping sample due to error: {e}")
        if max_samples is not None and sample_count >= max_samples:
            break

    summary = summarize(rows)

    if not summary:
        print("No samples evaluated. Check dataset paths.")
        return (summary, rows) if return_rows else summary

    if print_summary:
        print("\n" + "=" * 60)
        print(f"Perceptual & Timing Evaluation ({int(summary['num_samples'])} samples)")
        print("=" * 60)
        print(f"PSNR (LR->HR): {summary['psnr_lr']:.2f} dB | PSNR (SR->HR): {summary['psnr_sr']:.2f} dB | ΔPSNR: {summary['psnr_improve_db']:.2f} dB")
        print(f"SSIM (LR->HR): {summary['ssim_lr']:.4f} | SSIM (SR->HR): {summary['ssim_sr']:.4f} | ΔSSIM: {summary['ssim_improve']:.4f}")
        print(f"MSE (LR->HR): {summary['mse_lr']:.4f} | MSE (SR->HR): {summary['mse_sr']:.4f} | Artifact reduction (MSE ↓): {summary['mse_reduction_pct']:.2f}%")
        print(f"Sharpness (var Laplacian) LR: {summary['sharpness_lr']:.2f} | SR: {summary['sharpness_sr']:.2f} | Sharpness ↑: {summary['sharpness_improve_pct']:.2f}%")
        print(
            f"Generation time per image: mean {summary['inference_seconds']:.3f}s (p10 {summary['inference_seconds_p10']:.3f}s – p90 {summary['inference_seconds_p90']:.3f}s)"
        )

    # Optional CSV save
    if save_csv:
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(rows)
            df.to_csv(save_csv, index=False)
            print(f"Saved per-sample metrics to: {save_csv}")
        except Exception as e:
            print(f"Could not save CSV: {e}")

    return (summary, rows) if return_rows else summary


if __name__ == "__main__":
    run_perceptual_eval(max_samples=None)



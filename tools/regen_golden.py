#!/usr/bin/env python3
"""
Regenerate golden inputs/outputs and the manifest under tests/golden/.

Run in a **pinned** environment (see manifest ``reference_environment``), then review
the diff before committing:

  pip install 'numpy==…' 'opencv-python-headless==…'
  pip install -e .
  python tools/regen_golden.py
"""
from __future__ import annotations

import importlib.metadata as im
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "golden" / "data"
MANIFEST_PATH = ROOT / "tests" / "golden" / "manifest.json"

# Import package from the repo (editable install optional if PYTHONPATH=src)
sys.path.insert(0, str(ROOT / "src"))
from pykuwahara.kuwahara import kuwahara  # noqa: E402


def _checker_2d(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy + xx) % 2 * 255).astype(np.uint8)


def _gradient_2d(h: int, w: int) -> np.ndarray:
    row = np.linspace(0.0, 1.0, w, dtype=np.float32)
    return np.tile(row, (h, 1))


def _stripes_bgr(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for j in range(w):
        img[:, j] = (j * 7 % 256, j * 13 % 256, j * 3 % 256)
    return img


def _stripes_lab_l_channel() -> np.ndarray:
    bgr = _stripes_bgr(28, 28)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l_ch, _a, _b = cv2.split(lab)
    return l_ch


INPUT_BUILDERS = {
    "checker2d_u8.npy": lambda: _checker_2d(32, 32),
    "gradient2d_f32.npy": lambda: _gradient_2d(24, 32),
    "stripes3d_u8.npy": lambda: _stripes_bgr(28, 28),
    "stripes3d_u8_image2d_lab_l.npy": _stripes_lab_l_channel,
}


def _cases() -> list[dict]:
    return [
        {
            "id": "checker2d_u8_mean_r2",
            "input": "checker2d_u8.npy",
            "output": "checker2d_u8_mean_r2.npy",
            "method": "mean",
            "radius": 2,
            "sigma": None,
        },
        {
            "id": "checker2d_u8_gaussian_r2",
            "input": "checker2d_u8.npy",
            "output": "checker2d_u8_gaussian_r2.npy",
            "method": "gaussian",
            "radius": 2,
            "sigma": None,
        },
        {
            "id": "gradient2d_f32_mean_r2",
            "input": "gradient2d_f32.npy",
            "output": "gradient2d_f32_mean_r2.npy",
            "method": "mean",
            "radius": 2,
            "sigma": None,
        },
        {
            "id": "stripes3d_u8_mean_r2",
            "input": "stripes3d_u8.npy",
            "output": "stripes3d_u8_mean_r2.npy",
            "method": "mean",
            "radius": 2,
            "sigma": None,
        },
        {
            "id": "stripes3d_u8_gaussian_r2",
            "input": "stripes3d_u8.npy",
            "output": "stripes3d_u8_gaussian_r2.npy",
            "method": "gaussian",
            "radius": 2,
            "sigma": 1.0,
        },
        {
            "id": "stripes3d_u8_mean_r2_image2d_lab_l",
            "input": "stripes3d_u8.npy",
            "output": "stripes3d_u8_mean_r2_image2d_lab_l.npy",
            "method": "mean",
            "radius": 2,
            "sigma": None,
            "image_2d": "stripes3d_u8_image2d_lab_l.npy",
            "image_2d_source": "BGR2Lab_L",
        },
    ]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, builder in INPUT_BUILDERS.items():
        path = DATA_DIR / name
        arr = builder()
        np.save(path, arr, allow_pickle=False)
        print("wrote", path.relative_to(ROOT))

    cases = _cases()
    for case in cases:
        inp_path = DATA_DIR / case["input"]
        img = np.load(inp_path, allow_pickle=False)
        sigma = case["sigma"]
        kw = dict(method=case["method"], radius=case["radius"], sigma=sigma)
        if case.get("image_2d"):
            kw["image_2d"] = np.load(DATA_DIR / case["image_2d"], allow_pickle=False)
        out = kuwahara(img, **kw)
        out_path = DATA_DIR / case["output"]
        np.save(out_path, out, allow_pickle=False)
        print("wrote", out_path.relative_to(ROOT))

    try:
        opencv_ver = im.version("opencv-python-headless")
    except im.PackageNotFoundError:
        opencv_ver = cv2.__version__

    manifest = {
        "reference_environment": {
            "numpy": np.__version__,
            "opencv-python-headless": opencv_ver,
        },
        "notes": (
            "Regenerate with tools/regen_golden.py after pinning NumPy and "
            "opencv-python-headless to the versions above."
        ),
        "cases": cases,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote", MANIFEST_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()

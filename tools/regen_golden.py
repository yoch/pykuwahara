#!/usr/bin/env python3
"""
Régénère les entrées/sorties golden et le manifeste sous tests/golden/.

À exécuter dans un environnement aux versions **pinnées** (voir manifeste
`reference_environment`), puis valider le diff avant commit :

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

# Import package depuis le repo (pas besoin d’editable si PYTHONPATH=src)
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


INPUT_BUILDERS = {
    "checker2d_u8.npy": lambda: _checker_2d(32, 32),
    "gradient2d_f32.npy": lambda: _gradient_2d(24, 32),
    "stripes3d_u8.npy": lambda: _stripes_bgr(28, 28),
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
        out = kuwahara(
            img,
            method=case["method"],
            radius=case["radius"],
            sigma=sigma,
        )
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
            "Régénérer avec tools/regen_golden.py après avoir pinné NumPy et "
            "opencv-python-headless aux versions ci-dessus."
        ),
        "cases": cases,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote", MANIFEST_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()

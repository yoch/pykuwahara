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


# Gradient and stripes are constant along one axis: several quadrants would then have
# exactly equal variances, and the choice between them would depend on floating-point
# rounding (CPU / SIMD path). A seeded noise breaks these ties; seeds are chosen so that
# the check in ``_min_variance_gap`` passes with a comfortable margin.

# Minimum relative gap between the selected variance and any competing quadrant (with a
# different mean), relative to E[x²]: ~15 float32 ulps.
MIN_VARIANCE_GAP = 1e-6


def _gradient_2d(h: int, w: int) -> np.ndarray:
    row = np.linspace(0.0, 0.7, w, dtype=np.float32)
    noise = np.random.default_rng(1).random((h, w), dtype=np.float32) * 0.3
    return np.tile(row, (h, 1)) + noise


def _stripes_bgr(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.int16)
    for j in range(w):
        img[:, j] = (j * 7 % 256, j * 13 % 256, j * 3 % 256)
    img += np.random.default_rng(4).integers(-40, 41, size=img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


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


def _min_variance_gap(img: np.ndarray, guide: np.ndarray, method: str, radius: int, sigma) -> float:
    """
    Smallest relative gap (in float64) between the minimum quadrant variance and the variance
    of any other quadrant with a different mean, over all pixels (OpenCV REFLECT_101 borders).
    """
    r = radius
    if method == "mean":
        before = after = np.full(r + 1, 1.0 / (r + 1))
    else:
        g = cv2.getGaussianKernel(2 * r + 1, -1 if sigma is None else sigma, ktype=cv2.CV_64F).ravel()
        before, after = g[: r + 1] / g[: r + 1].sum(), g[r:] / g[r:].sum()
    halves = [(0, before), (r, after)]   # row/col offsets in the padded arrays
    img = np.pad(img.astype(np.float64), [(r, r), (r, r)] + [(0, 0)] * (img.ndim - 2), mode="reflect")
    guide = np.pad(guide.astype(np.float64), r, mode="reflect")
    worst = np.inf
    for i in range(guide.shape[0] - 2 * r):
        for j in range(guide.shape[1] - 2 * r):
            quadrants = []
            for di, wy in halves:
                for dj, wx in halves:
                    weights = np.outer(wy, wx)
                    rows, cols = slice(i + di, i + di + r + 1), slice(j + dj, j + dj + r + 1)
                    e2 = np.sum(weights * guide[rows, cols] ** 2)
                    var = e2 - np.sum(weights * guide[rows, cols]) ** 2
                    mean = np.tensordot(weights, img[rows, cols], axes=([0, 1], [0, 1]))
                    quadrants.append((var, mean, e2))
            quadrants.sort(key=lambda q: q[0])
            v0, m0, e0 = quadrants[0]
            for var, mean, _ in quadrants[1:]:
                if not np.allclose(mean, m0, rtol=0, atol=1e-9):
                    worst = min(worst, (var - v0) / max(e0, 1e-12))
    return worst


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
        if "image_2d" in kw:
            guide = kw["image_2d"]
        else:
            guide = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gap = _min_variance_gap(img, guide, case["method"], case["radius"], sigma)
        if gap < MIN_VARIANCE_GAP:
            raise SystemExit(
                f"{case['id']}: quadrant variances too close ({gap:.1e} < {MIN_VARIANCE_GAP:.0e}); "
                "the golden output would depend on floating-point rounding. Change the input."
            )
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

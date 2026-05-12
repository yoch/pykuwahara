"""
Régression golden + métriques sur ROI intérieure (hors bande de ``radius``).

Seuils (documentés) :
  - SSIM sur la ROI : >= 0.995 (identique à la référence pour l’environnement pinné).
  - PSNR sur la ROI : >= 40 dB (uint8 : data_range=255 ; float32 : data_range=1.0).

Mode ``GOLDEN_STRICT=1`` (job CI golden) : égalité stricte sur l’image entière
(uint8) ou ``allclose`` (float32) en plus des métriques.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

_skimage_metrics = pytest.importorskip("skimage.metrics", reason="extra [golden] : pip install -e '.[golden]'")
structural_similarity = _skimage_metrics.structural_similarity

from pykuwahara import kuwahara

pytestmark = pytest.mark.golden

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
DATA_DIR = GOLDEN_DIR / "data"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
_MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
_GOLDEN_CASES = _MANIFEST["cases"]

# Seuils ROI (intérieur, exclut ``radius`` pixels de bord sur chaque côté).
SSIM_MIN_ROI = 0.995
PSNR_MIN_ROI = 40.0


def _interior_roi(arr: np.ndarray, radius: int) -> np.ndarray:
    r = radius
    return arr[r : arr.shape[0] - r, r : arr.shape[1] - r]


def _psnr_roi(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    mse = np.mean((a64 - b64) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def _ssim_roi(
    expected: np.ndarray,
    actual: np.ndarray,
    data_range: float,
) -> float:
    kwargs: dict = {"data_range": data_range}
    if expected.ndim == 3:
        kwargs["channel_axis"] = -1
    return float(structural_similarity(expected, actual, **kwargs))


def _save_failure(case_id: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if os.environ.get("CI") != "true":
        return
    out_dir = Path("golden_failure")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / f"{case_id}_actual.npy", actual)
    np.save(out_dir / f"{case_id}_expected.npy", expected)
    np.save(out_dir / f"{case_id}_absdiff.npy", np.abs(actual.astype(np.float64) - expected.astype(np.float64)))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _MANIFEST


def test_manifest_has_reference_environment(manifest: dict) -> None:
    ref = manifest.get("reference_environment")
    assert isinstance(ref, dict)
    assert "numpy" in ref
    assert "opencv-python-headless" in ref


@pytest.mark.parametrize("case", _GOLDEN_CASES, ids=lambda c: c["id"])
def test_golden_case_ssim_psnr_and_optional_strict(case: dict) -> None:
    cid = case["id"]
    img = np.load(DATA_DIR / case["input"], allow_pickle=False)
    expected = np.load(DATA_DIR / case["output"], allow_pickle=False)
    radius = int(case["radius"])
    sigma = case.get("sigma")
    actual = kuwahara(
        img,
        method=case["method"],
        radius=radius,
        sigma=sigma,
    )

    data_range = 255.0 if actual.dtype == np.uint8 else 1.0
    roi_e = _interior_roi(expected, radius)
    roi_a = _interior_roi(actual, radius)
    assert roi_e.shape == roi_a.shape

    ssim = _ssim_roi(roi_e, roi_a, data_range=data_range)
    psnr = _psnr_roi(roi_e, roi_a, data_range=data_range)

    try:
        assert ssim >= SSIM_MIN_ROI, f"{cid}: SSIM ROI {ssim} < {SSIM_MIN_ROI}"
        assert psnr >= PSNR_MIN_ROI, f"{cid}: PSNR ROI {psnr} < {PSNR_MIN_ROI}"
        if os.environ.get("GOLDEN_STRICT") == "1":
            if actual.dtype == np.uint8:
                assert np.array_equal(actual, expected), cid
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, err_msg=cid)
    except AssertionError:
        _save_failure(cid, actual, expected)
        raise

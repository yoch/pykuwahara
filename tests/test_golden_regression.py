"""
Golden regression + metrics on the interior ROI (strip of ``radius`` pixels excluded).

**Scope**: classic Kuwahara only (``method`` mean / gaussian, same pipeline as
``tools/regen_golden.py``). A future anisotropic filter should use its own manifest
and test module if needed.

Documented thresholds:
  - SSIM on ROI: >= 0.995 (matches pinned reference environment).
  - PSNR on ROI: >= 40 dB (uint8: data_range=255 ; float32: data_range=1.0).

``GOLDEN_STRICT=1`` (golden CI job): strict equality on the full image (uint8) or
``allclose`` (float32) in addition to the metrics.

Noise degradation — prove SSIM reacts to a **local** perturbation:

  PYKUWAHARA_RUN_NOISE_CHECK=1 python -m pytest tests/test_golden_regression.py -k degradation_noise -v

Cases corrupt an 8×8 block (inside the ROI); a single pixel can leave global SSIM
above ``SSIM_MIN_ROI``.

Requires the ``[golden]`` extra (scikit-image). ``degradation_noise`` tests are skipped
by default to keep CI light.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

_skimage_metrics = pytest.importorskip("skimage.metrics", reason="install with: pip install -e '.[golden]'")
structural_similarity = _skimage_metrics.structural_similarity

from pykuwahara import kuwahara

pytestmark = pytest.mark.golden

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
DATA_DIR = GOLDEN_DIR / "data"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
_MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
_GOLDEN_CASES = _MANIFEST["cases"]

# ROI thresholds (interior excludes ``radius`` pixels on each side).
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
    kw = dict(method=case["method"], radius=radius, sigma=sigma)
    if case.get("image_2d"):
        kw["image_2d"] = np.load(DATA_DIR / case["image_2d"], allow_pickle=False)
    actual = kuwahara(img, **kw)

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


@pytest.mark.skipif(
    not os.environ.get("PYKUWAHARA_RUN_NOISE_CHECK"),
    reason="Set PYKUWAHARA_RUN_NOISE_CHECK=1 to verify SSIM sensitivity to noise on the expected array.",
)
def test_degradation_noise_region_on_expected_drops_ssim_below_min() -> None:
    """8×8 corruption on the expected array: ROI SSIM must drop below SSIM_MIN_ROI."""
    case = next(c for c in _GOLDEN_CASES if c["id"] == "stripes3d_u8_mean_r2")
    img = np.load(DATA_DIR / case["input"], allow_pickle=False)
    expected = np.load(DATA_DIR / case["output"], allow_pickle=False)
    radius = int(case["radius"])
    actual = kuwahara(img, method=case["method"], radius=radius, sigma=case.get("sigma"))
    data_range = 255.0
    roi_a = _interior_roi(actual, radius)
    roi_e = _interior_roi(expected, radius)
    ssim_clean = _ssim_roi(roi_e, roi_a, data_range=data_range)
    assert ssim_clean >= SSIM_MIN_ROI

    corrupted = expected.copy()
    ci, cj = corrupted.shape[0] // 2, corrupted.shape[1] // 2
    sl = slice(ci - 4, ci + 4)
    sc = slice(cj - 4, cj + 4)
    corrupted[sl, sc, :] = np.uint8(255) - corrupted[sl, sc, :]
    roi_c = _interior_roi(corrupted, radius)
    ssim_noisy = _ssim_roi(roi_c, roi_a, data_range=data_range)
    assert ssim_noisy < SSIM_MIN_ROI, (
        f"Noisy SSIM {ssim_noisy} should be < {SSIM_MIN_ROI}; golden tests did not react to noise."
    )


@pytest.mark.skipif(
    not os.environ.get("PYKUWAHARA_RUN_NOISE_CHECK"),
    reason="Set PYKUWAHARA_RUN_NOISE_CHECK=1 to verify SSIM sensitivity to noisy input.",
)
def test_degradation_noise_region_on_input_drops_ssim_against_golden_expected() -> None:
    """8×8 corruption on the input: output must no longer match golden SSIM."""
    case = next(c for c in _GOLDEN_CASES if c["id"] == "stripes3d_u8_mean_r2")
    img = np.load(DATA_DIR / case["input"], allow_pickle=False)
    expected = np.load(DATA_DIR / case["output"], allow_pickle=False)
    radius = int(case["radius"])
    img_noisy = img.copy()
    ii, jj = img_noisy.shape[0] // 2, img_noisy.shape[1] // 2
    sl = slice(ii - 4, ii + 4)
    sc = slice(jj - 4, jj + 4)
    img_noisy[sl, sc, :] = np.uint8(255) - img_noisy[sl, sc, :]
    actual = kuwahara(img_noisy, method=case["method"], radius=radius, sigma=case.get("sigma"))
    data_range = 255.0
    roi_a = _interior_roi(actual, radius)
    roi_e = _interior_roi(expected, radius)
    ssim_perturbed = _ssim_roi(roi_e, roi_a, data_range=data_range)
    assert ssim_perturbed < SSIM_MIN_ROI, (
        f"SSIM {ssim_perturbed} should be < {SSIM_MIN_ROI} after noisy input."
    )

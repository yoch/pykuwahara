"""
Tests for ``kuwahara`` (classic four-quadrant Kuwahara filter).

Naming convention (KISS, single file):

- ``test_filter_contract_*``: generic contract — any reasonable ndarray filter should
  match the ``shape`` / ``dtype`` and argument errors exercised here. Reuse or copy
  thoughtfully for future algorithms.

- ``test_classic_kuwahara_quadrant_*``: **Kuwahara-specific** properties (four overlapping
  subwindows, minimum-variance choice). Do not assume an anisotropic or other filter
  satisfies them without review.

Controlled degradation (prove the tests "bite"):

- Targeted implementation mutations: set ``PYKUWAHARA_RUN_MUTATION_CHECK=1`` then run
  pytest on this file; the tests at the end **pass** only if, under patch, quadrant
  bounds are actually violated (otherwise regression detection is too weak).

  Example::

      PYKUWAHARA_RUN_MUTATION_CHECK=1 python -m pytest tests/test_kuwahara.py -k degradation_mutation -v

- Noise / golden: see ``tests/test_golden_regression.py`` and ``PYKUWAHARA_RUN_NOISE_CHECK=1``.

Manual checklist (throwaway branch, **do not merge**) in ``src/pykuwahara/kuwahara.py``:

- Replace ``np.argmin(stddevs, axis=0)`` with ``np.argmax``: tests
  ``classic_kuwahara_quadrant_output_within_local_window_bounds`` may **still pass**
  (any quadrant mean stays inside the patch min/max envelope); **golden** may fail.
  To break bounds, use the automated ``degradation_mutation_biased_output_*`` test or
  a manual bias on the output.

- Add a small bias before ``astype(orig_img.dtype)``: expect golden / SSIM failure;
  local bounds may still pass if the bias stays inside the envelope.
"""
from __future__ import annotations

import os
import warnings

import importlib

import cv2
import numpy as np
import pytest

from pykuwahara import kuwahara

_kuwahara_module = importlib.import_module("pykuwahara.kuwahara")

# Reproducible: same sequence on every run (avoids implicit global np.random).
_RNG = np.random.default_rng(42)


def random_image_1channel_uint():
    return _RNG.integers(0, 255, size=(128, 128), dtype=np.ubyte)


def random_image_3channels_uint():
    return _RNG.integers(0, 255, size=(128, 128, 3), dtype=np.ubyte)


def random_image_1channel_float():
    return _RNG.random((128, 128), dtype=np.float32)


def random_image_3channels_float():
    return _RNG.random((128, 128, 3), dtype=np.float32)


img_1c_uint = random_image_1channel_uint()
img_3c_uint = random_image_3channels_uint()
img_1c_float = random_image_1channel_float()
img_3c_float = random_image_3channels_float()
images = [img_1c_uint, img_3c_uint, img_1c_float, img_3c_float]
images_1c = [img_1c_uint, img_1c_float]
images_3c = [img_3c_uint, img_3c_float]


def _quadrant_local_bounds_satisfied(img: np.ndarray, out: np.ndarray, radius: int) -> bool:
    """True if, on the interior, each channel of ``out`` lies in the patch (2r+1)² min/max."""
    h, w = img.shape[:2]
    r = radius
    patch_lo = np.empty_like(out[r : h - r, r : w - r])
    patch_hi = np.empty_like(out[r : h - r, r : w - r])
    for i in range(r, h - r):
        for j in range(r, w - r):
            patch = img[i - r : i + r + 1, j - r : j + r + 1]
            patch_lo[i - r, j - r] = patch.min(axis=(0, 1))
            patch_hi[i - r, j - r] = patch.max(axis=(0, 1))
    inner = out[r : h - r, r : w - r]
    return bool(np.all(inner >= patch_lo) and np.all(inner <= patch_hi))


# ---------------------------------------------------------------------------
# Generic contract (shape / dtype / errors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image", images)
@pytest.mark.parametrize("radius", [1, 2, 3, 4, 5])
def test_filter_contract_kuwahara_mean_preserves_shape_and_dtype(image, radius):
    filtered = kuwahara(image, method="mean", radius=radius)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype


@pytest.mark.parametrize("image", images)
@pytest.mark.parametrize("radius,sigma", [(1, None), (2, 0.8), (3, 1.2), (4, None), (5, 1.5)])
def test_filter_contract_kuwahara_gaussian_preserves_shape_and_dtype(image, radius, sigma):
    filtered = kuwahara(image, method="gaussian", radius=radius, sigma=sigma)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype


@pytest.mark.parametrize("image", images_3c)
@pytest.mark.parametrize("method", ["gaussian", "mean"])
@pytest.mark.parametrize("grayconv", [cv2.COLOR_BGR2GRAY, cv2.COLOR_RGB2GRAY])
def test_filter_contract_kuwahara_grayconv_preserves_shape_and_dtype(image, method, grayconv):
    filtered = kuwahara(image, method=method, grayconv=grayconv)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype


@pytest.mark.parametrize("image", images_3c)
@pytest.mark.parametrize("method", ["gaussian", "mean"])
def test_filter_contract_kuwahara_with_image_2d_preserves_shape_and_dtype(image, method):
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    filtered = kuwahara(image, method=method, image_2d=v)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype


def test_filter_contract_kuwahara_float64_3d_grayconv_warns_and_runs():
    """
    3D float64 without ``image_2d``: OpenCV rejects CV_64F for ``cvtColor``; the code
    converts to float32 for grayscale only and emits ``UserWarning``.
    """
    img = np.full((10, 12, 3), 0.5, dtype=np.float64)
    with pytest.warns(UserWarning, match="float64"):
        out = kuwahara(img, method="mean", radius=2)
    assert out.shape == img.shape
    assert out.dtype == np.float64

    g2 = np.full((10, 12), 0.3, dtype=np.float64)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out2 = kuwahara(img, method="mean", radius=2, image_2d=g2)
    assert not any("float64" in str(w.message) for w in rec)
    assert out2.shape == img.shape
    assert out2.dtype == np.float64


@pytest.mark.parametrize("method", ["mean", "gaussian"])
def test_filter_contract_kuwahara_float64_2d_ok_without_image_2d(method):
    img = np.full((14, 15), 0.25, dtype=np.float64)
    sigma = 0.8 if method == "gaussian" else None
    out = kuwahara(img, method=method, radius=2, sigma=sigma)
    assert out.shape == img.shape
    assert out.dtype == np.float64


@pytest.mark.parametrize("image", images[:1])
@pytest.mark.parametrize("method", ["gaussian", "mean"])
def test_filter_contract_kuwahara_radius_must_be_int(image, method):
    with pytest.raises(TypeError):
        kuwahara(image, method=method, radius=1.5)


@pytest.mark.parametrize("image", images[:1])
@pytest.mark.parametrize("method", ["gaussian", "mean"])
@pytest.mark.parametrize("radius", [-1, 0])
def test_filter_contract_kuwahara_radius_must_be_positive(image, method, radius):
    with pytest.raises(ValueError):
        kuwahara(image, method=method, radius=radius)


@pytest.mark.parametrize("image", images[:1])
@pytest.mark.parametrize("method", ["gaussian", "mean"])
def test_filter_contract_kuwahara_rejects_bad_ndim(image, method):
    with pytest.raises(TypeError):
        kuwahara(image.reshape(-1), method=method)


@pytest.mark.parametrize("image", images[:1])
def test_filter_contract_kuwahara_rejects_unknown_method(image):
    with pytest.raises(NotImplementedError):
        kuwahara(image, method="nonexistent")


# ---------------------------------------------------------------------------
# Classic four-quadrant Kuwahara (semantic properties)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mean", "gaussian"])
@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_classic_kuwahara_quadrant_constant_image_unchanged_2d(method, radius, dtype):
    """Constant 2D image: output equals input (equal quadrant variances)."""
    val = 0.42 if dtype == np.float32 else 107
    img = np.full((24, 24), val, dtype=dtype)
    out = kuwahara(img, method=method, radius=radius)
    if dtype == np.float32:
        np.testing.assert_allclose(out, img, rtol=1e-5, atol=1e-5)
    else:
        assert np.array_equal(out, img)


@pytest.mark.parametrize("method", ["mean", "gaussian"])
@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_classic_kuwahara_quadrant_constant_image_unchanged_3d(method, radius, dtype):
    """Constant 3-channel image: same as 2D case."""
    if dtype == np.float32:
        img = np.full((20, 20, 3), (0.1, 0.5, 0.9), dtype=dtype)
    else:
        img = np.full((20, 20, 3), (10, 120, 200), dtype=np.uint8)
    out = kuwahara(img, method=method, radius=radius)
    if dtype == np.float32:
        np.testing.assert_allclose(out, img, rtol=1e-5, atol=1e-5)
    else:
        assert np.array_equal(out, img)


@pytest.mark.parametrize("method", ["mean", "gaussian"])
@pytest.mark.parametrize("radius", [2, 3])
def test_classic_kuwahara_quadrant_output_within_local_window_bounds(method, radius):
    """
    Interior (strip of ``radius`` from edges): each channel lies in the
    (2*radius+1)² patch min/max (output is a quadrant mean).
    """
    h, w = 24, 28
    rng = np.random.default_rng(12345)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    out = kuwahara(img, method=method, radius=radius)
    assert _quadrant_local_bounds_satisfied(img, out, radius)


# ---------------------------------------------------------------------------
# Controlled degradation — mutations (see module docstring)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("PYKUWAHARA_RUN_MUTATION_CHECK"),
    reason="Set PYKUWAHARA_RUN_MUTATION_CHECK=1 to verify mutations break quadrant bounds.",
)
def test_degradation_mutation_biased_output_breaks_quadrant_bounds(monkeypatch):
    """
    Large post-filter bias: generally violates patch min/max envelope.

    Note: swapping ``argmin`` for ``argmax`` is **not** enough — both still pick a
    quadrant mean, so the patch min/max property may still hold.
    """
    _orig_k = _kuwahara_module.kuwahara

    def _biased(orig_img, *args, **kwargs):
        out = _orig_k(orig_img, *args, **kwargs)
        return np.clip(out.astype(np.int16) + 120, 0, 255).astype(out.dtype)

    monkeypatch.setattr(_kuwahara_module, "kuwahara", _biased, raising=False)
    h, w = 24, 28
    radius = 2
    rng = np.random.default_rng(12345)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    out = _kuwahara_module.kuwahara(img, method="mean", radius=radius)
    assert not _quadrant_local_bounds_satisfied(img, out, radius), (
        "Artificial bias did not violate bounds: degradation test proves nothing."
    )


@pytest.mark.skipif(
    not os.environ.get("PYKUWAHARA_RUN_MUTATION_CHECK"),
    reason="Set PYKUWAHARA_RUN_MUTATION_CHECK=1 to verify mutations break the constant invariant.",
)
def test_degradation_mutation_zero_output_breaks_constant_invariant(monkeypatch):
    """Force zero output: a non-zero constant input must no longer be preserved."""
    _orig_k = _kuwahara_module.kuwahara

    def _zeros(orig_img, *args, **kwargs):
        return np.zeros_like(orig_img)

    monkeypatch.setattr(_kuwahara_module, "kuwahara", _zeros, raising=False)
    img = np.full((16, 16, 3), (40, 80, 120), dtype=np.uint8)
    out = _kuwahara_module.kuwahara(img, method="mean", radius=2)
    assert not np.array_equal(out, img), (
        "Zero output did not break expected equality: degradation test proves nothing."
    )

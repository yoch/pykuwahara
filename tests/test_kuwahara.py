import pytest
import numpy as np
import cv2
from pykuwahara import kuwahara

# Reproductible: même séquence sur toutes les exécutions (évite np.random global implicite).
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


####################    TESTS    ####################

@pytest.mark.parametrize("image", images)
@pytest.mark.parametrize("radius", [1, 2, 3, 4, 5])
def test_kuwahara_mean(image, radius):
    filtered = kuwahara(image, method='mean', radius=radius)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype


@pytest.mark.parametrize("image", images)
@pytest.mark.parametrize("radius,sigma", [(1, None), (2, 0.8), (3, 1.2), (4, None), (5, 1.5)])
def test_kuwahara_gaussian(image, radius, sigma):
    filtered = kuwahara(image, method='gaussian', radius=radius, sigma=sigma)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype


@pytest.mark.parametrize("image", images_3c)
@pytest.mark.parametrize("method", ['gaussian', 'mean'])
@pytest.mark.parametrize("grayconv", [cv2.COLOR_BGR2GRAY, cv2.COLOR_RGB2GRAY])
def test_kuwahara_grayconv(image, method, grayconv):
    filtered = kuwahara(image, method=method, grayconv=grayconv)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype


@pytest.mark.parametrize("image", images_3c)
@pytest.mark.parametrize("method", ['gaussian', 'mean'])
def test_kuwahara_cvtcolor(image, method):
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    filtered = kuwahara(image, method=method, image_2d=v)
    assert filtered.shape == image.shape
    assert filtered.dtype == image.dtype


#################### TESTS PROPRIÉTÉS ####################

@pytest.mark.parametrize("method", ['mean', 'gaussian'])
@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_constant_image_unchanged_2d(method, radius, dtype):
    """Image constante 2D : la sortie doit coïncider avec l’entrée (Kuwahara = moyenne des quartiers identiques)."""
    val = 0.42 if dtype == np.float32 else 107
    img = np.full((24, 24), val, dtype=dtype)
    out = kuwahara(img, method=method, radius=radius)
    if dtype == np.float32:
        np.testing.assert_allclose(out, img, rtol=1e-5, atol=1e-5)
    else:
        assert np.array_equal(out, img)


@pytest.mark.parametrize("method", ['mean', 'gaussian'])
@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_constant_image_unchanged_3d(method, radius, dtype):
    """Image constante 3 canaux : idem."""
    if dtype == np.float32:
        img = np.full((20, 20, 3), (0.1, 0.5, 0.9), dtype=dtype)
    else:
        img = np.full((20, 20, 3), (10, 120, 200), dtype=np.uint8)
    out = kuwahara(img, method=method, radius=radius)
    if dtype == np.float32:
        np.testing.assert_allclose(out, img, rtol=1e-5, atol=1e-5)
    else:
        assert np.array_equal(out, img)


@pytest.mark.parametrize("method", ['mean', 'gaussian'])
@pytest.mark.parametrize("radius", [2, 3])
def test_output_within_local_window_bounds(method, radius):
    """
    Sur l’intérieur (hors bande de radius), chaque canal de sortie est borné
    par le min/max du voisinage (2*radius+1)^2 (la sortie est une moyenne d’un quartier).
    """
    h, w = 24, 28
    rng = np.random.default_rng(12345)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    out = kuwahara(img, method=method, radius=radius)
    r = radius
    patch_lo = np.empty_like(out[r:h - r, r:w - r])
    patch_hi = np.empty_like(out[r:h - r, r:w - r])
    for i in range(r, h - r):
        for j in range(r, w - r):
            patch = img[i - r:i + r + 1, j - r:j + r + 1]
            patch_lo[i - r, j - r] = patch.min(axis=(0, 1))
            patch_hi[i - r, j - r] = patch.max(axis=(0, 1))
    inner = out[r:h - r, r:w - r]
    assert np.all(inner >= patch_lo)
    assert np.all(inner <= patch_hi)


#################### TESTS ERRORS ####################

@pytest.mark.parametrize("image", images[:1])
@pytest.mark.parametrize("method", ['gaussian', 'mean'])
def test_kuwahara_gaussian_radius_float(image, method):
    with pytest.raises(TypeError):
        kuwahara(image, method=method, radius=1.5)


@pytest.mark.parametrize("image", images[:1])
@pytest.mark.parametrize("method", ['gaussian', 'mean'])
@pytest.mark.parametrize("radius", [-1, 0])
def test_kuwahara_gaussian_radius_neg_or_zero(image, method, radius):
    with pytest.raises(ValueError):
        kuwahara(image, method=method, radius=radius)


@pytest.mark.parametrize("image", images[:1])
@pytest.mark.parametrize("method", ['gaussian', 'mean'])
def test_kuwahara_gaussian_bad_shape(image, method):
    with pytest.raises(TypeError):
        kuwahara(image.reshape(-1), method=method)


@pytest.mark.parametrize("image", images[:1])
def test_kuwahara_gaussian_bad_method(image):
    with pytest.raises(NotImplementedError):
        kuwahara(image, method='inexistant')

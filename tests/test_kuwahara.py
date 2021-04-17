import pytest
import numpy as np
import cv2
from pykuwahara import kuwahara


def random_image_1channel_uint():
    return np.random.randint(0, 255, size=(128, 128), dtype=np.ubyte)

def random_image_3channels_uint():
    return np.random.randint(0, 255, size=(128, 128, 3), dtype=np.ubyte)

def random_image_1channel_float():
    return np.random.rand(128, 128).astype(np.float32)

def random_image_3channels_float():
    return np.random.rand(128, 128, 3).astype(np.float32)

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

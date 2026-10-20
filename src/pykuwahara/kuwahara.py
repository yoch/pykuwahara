from typing import Literal, Optional
import numbers
import warnings

import cv2
import numpy as np


# dtypes accepted as-is by cv2.cvtColor (native byte order only)
_CVTCOLOR_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32))


def _work_dtype(dtype: np.dtype) -> type:
    """Floating dtype used for the computations (float32 unless it loses precision)."""
    if dtype.kind in 'biu' and dtype.itemsize == 1:
        return np.float32
    if dtype.kind == 'f' and dtype.itemsize <= 4:
        return np.float32
    return np.float64


def kuwahara(
    orig_img: np.ndarray,
    method: Literal["mean", "gaussian"] = "mean",
    radius: int = 3,
    sigma: Optional[float] = None,
    grayconv: int = cv2.COLOR_BGR2GRAY,
    image_2d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Get the image filtered with Kuwahara method.

    Computations are done in float32 for 8-bit, float16 and float32 inputs, and in
    float64 for all other dtypes (16-bit and wider integers, float64). For integer
    inputs, the output is truncated toward zero when cast back to the input dtype.

    :param orig_img: original numpy image, of shape ``(H, W)`` or ``(H, W, C)``
    :type orig_img: :class:`numpy.ndarray`
    :param method: method used to compute the pixels values
    :type method: `Literal['mean', 'gaussian']`
    :param radius: the window radius (`winsize = 2 * radius + 1`), greater than or equal to 1
    :type radius: `int`
    :param sigma: the sigma used if method is "gaussian", automatically computed by OpenCV when `None`
    :type sigma: `float` or `None`
    :param grayconv: OpenCV color conversion code for ``cv2.cvtColor(orig_img, grayconv)`` when ``image_2d``
        is omitted (3-channel input). It must produce a single-channel image. Default is ``cv2.COLOR_BGR2GRAY``.
    :type grayconv: `int`
    :param image_2d: the 1-channel image of shape ``(H, W)`` used to compute the variance, used instead
        of `grayconv` (3-channel input) or of the image itself (2-D input)
    :type image_2d: `numpy.ndarray` or `None`
    :returns: the filtered image, with the same shape and dtype as `orig_img`
    :rtype: `numpy.ndarray`
    :raises TypeError: if `orig_img` is not 2-D or 3-D, or if `radius` is not an integer
    :raises ValueError: if `radius` < 1, if `method` is unknown, if `image_2d` has not the
        shape ``(H, W)``, or if `grayconv` does not produce a single-channel image
    """

    # Filter description:
    # The Kuwahara filter works on a window divided into 4 overlapping
    # subwindows (for a 5x5 pixels example, see below). In each subwindow, the mean and
    # variance are computed. The output value (located at the center of the
    # window) is set to the mean of the subwindow with the smallest variance.
    #
    #     ( a  a  ab   b  b)
    #     ( a  a  ab   b  b)
    #     (ac ac abcd bd bd)
    #     ( c  c  cd   d  d)
    #     ( c  c  cd   d  d)
    #
    # Inspired by:
    #       python-kuwahara by Andrew Dussault, 2015 : https://github.com/adussault/python-kuwahara
    #       which it itself based on original Matlab code from Luca Balbi, 2007

    if orig_img.ndim != 2 and orig_img.ndim != 3:
        raise TypeError("Incorrect number of dimensions (expected 2 or 3)")

    if isinstance(radius, bool) or not isinstance(radius, numbers.Integral):
        raise TypeError('`radius` must be int')
    radius = int(radius)

    if radius < 1:
        raise ValueError('`radius` must be greater than or equal to 1')

    if method not in ('mean', 'gaussian'):
        raise ValueError('unsupported method %s' % method)

    if method == 'gaussian' and sigma is None:
        sigma = -1
        # then computed by OpenCV as : 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        # see: https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa

    # convert to float if necessary for further math computation
    work_dtype = _work_dtype(orig_img.dtype)
    image = orig_img.astype(work_dtype, copy=False)

    if image_2d is not None:
        image_2d = np.asarray(image_2d)
        if image_2d.ndim != 2 or image_2d.shape != image.shape[:2]:
            raise ValueError(
                f"`image_2d` must be a 2-D array of shape {image.shape[:2]}, got {image_2d.shape}"
            )
        image_2d = image_2d.astype(work_dtype, copy=False)

    # preallocate these arrays
    avgs = np.empty((4, *image.shape), dtype=work_dtype)
    variances = np.empty((4, *image.shape[:2]), dtype=work_dtype)

    if image.ndim == 3 and image_2d is None:
        if orig_img.dtype in _CVTCOLOR_DTYPES and orig_img.dtype.isnative:
            cvt_src = orig_img
        else:
            # OpenCV rejects other depths (e.g. CV_16S, CV_64F) for typical color conversions.
            if orig_img.dtype.kind == 'f' and orig_img.dtype.itemsize == 8:
                warnings.warn(
                    "3-channel float64 input without `image_2d`: OpenCV `cv2.cvtColor` does not "
                    "support CV_64F; the grayscale conversion (grayconv) is computed in float32, "
                    "the filtering itself stays in float64. Pass an explicit 1-channel `image_2d` "
                    "for a full-precision variance guide.",
                    UserWarning,
                    stacklevel=2,
                )
            cvt_src = image.astype(np.float32, copy=False)
        image_2d = cv2.cvtColor(cvt_src, grayconv)
        if image_2d.ndim != 2:
            raise ValueError("`grayconv` must produce a single-channel (2-D) image")
        image_2d = image_2d.astype(work_dtype, copy=False)

    if image_2d is None:
        image_2d = image
        avgs_2d = avgs
    else:
        avgs_2d = np.empty((4, *image.shape[:2]), dtype=work_dtype)

    # Create a pixel-by-pixel square of the image
    squared_img = image_2d ** 2

    if method == 'mean':
        kxy = np.ones(radius + 1, dtype=work_dtype) / (radius + 1)    # kernelX and kernelY (same)
    elif method == 'gaussian':
        ktype = cv2.CV_64F if work_dtype is np.float64 else cv2.CV_32F
        kxy = cv2.getGaussianKernel(2 * radius + 1, sigma, ktype=ktype)
        kxy /= kxy[radius:].sum()   # normalize the semi-kernels
        klr = np.array([kxy[:radius+1], kxy[radius:]])
        kindexes = [[1, 1], [1, 0], [0, 1], [0, 0]]

    # the anchor (pixel position) for all kernel quadrants
    anchors = [(0, 0), (0,  radius), (radius, 0), (radius, radius)]

    # Calculation of averages and variances on subwindows
    for k in range(4):
        if method == 'mean':
            kx = ky = kxy
        elif method == 'gaussian':
            kx, ky = klr[kindexes[k]]
        cv2.sepFilter2D(image, -1, kx, ky, avgs[k], anchors[k])
        if avgs_2d is not avgs: # else, this is already done...
            cv2.sepFilter2D(image_2d, -1, kx, ky, avgs_2d[k], anchors[k])
        cv2.sepFilter2D(squared_img, -1, kx, ky, variances[k], anchors[k])
        variances[k] = variances[k] - avgs_2d[k] ** 2    # compute the final variance on subwindow

    # Choice of index with minimum variance
    indices = np.argmin(variances, axis=0)

    # Building the filtered image
    if image.ndim == 2:
        filtered = np.take_along_axis(avgs, indices[None,...], 0).reshape(image.shape)
    else:   # then avgs.ndim == 4
        filtered = np.take_along_axis(avgs, indices[None,...,None], 0).reshape(image.shape)

    return filtered.astype(orig_img.dtype)

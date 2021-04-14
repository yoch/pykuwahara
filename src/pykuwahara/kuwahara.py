import numpy as np
import cv2

def kuwahara(orig_img, radius):
    """
    Get the image filtered with Kuwahara method.

    :param orig_img: original numpy image (support multichannel)
    :type orig_img: :class:`numpy.ndarray`
    :param radius: the window radius (winsize = 2 * radius + 1)
    :type radius: `int`
    :returns: the filtered image
    :rtype: `numpy.ndarray`
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

    if not isinstance(radius, int):
        raise TypeError('radius must be int')

    if radius < 1:
        raise ValueError('radius must be greater or equal 1')

    # convert to float64 if necessary for further math computation
    image = orig_img.astype(np.float64, copy=False)

    # Create a pixel-by-pixel square of the image
    squared_img = image ** 2

    # preallocate these arrays
    avgs = np.empty((4, *image.shape))
    stddevs = np.empty((4, *image.shape))

    kxy = np.ones(radius + 1) / (radius + 1)    # kernelX and kernelY (same)
    shift = [(0, 0), (0,  radius), (radius, 0), (radius, radius)]

    # Calculation of averages and variances on subwindows
    for k in range(4):
        cv2.sepFilter2D(image, -1, kxy, kxy, avgs[k], shift[k])
        cv2.sepFilter2D(squared_img, -1, kxy, kxy, stddevs[k], shift[k])
        stddevs[k] = stddevs[k] - avgs[k] ** 2    # variance on subwindow

    # Choice of index with minimum variance
    indices = np.argmin(stddevs, axis=0)

    # Building the filtered image
    filtered = np.take_along_axis(avgs, indices[None,...], 0).reshape(image.shape)

    return filtered.astype(orig_img.dtype)

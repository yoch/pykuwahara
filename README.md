# pykuwahara

Kuwahara filter in Python (numpy + OpenCV).

> The Kuwahara filter is a non-linear smoothing filter used in image processing for adaptive noise reduction. It is able to apply smoothing on the image while preserving the edges.
> Source: [Wikipedia](https://en.wikipedia.org/wiki/Kuwahara_filter)

This implementation provide two variants of the filter:
- The classic one, using a uniform kernel to compute the window mean.
- A gaussian based filter, by computing the window gaussian mean. This is inspired by the [ImageMagick](http://www.fmwconcepts.com/imagemagick/kuwahara/index.php) approach.

## Installation

`pip install pykuwahara`

## Usage

```
import cv2
from pykuwahara import kuwahara

image = cv2.imread('lena.jpg')

filtered1 = kuwahara(image, radius=3, method='mean')    # default
filtered2 = kuwahara(image, radius=3, method='gaussian')

cv2.imwrite('lena-filtered1.jpg', filtered1)
cv2.imwrite('lena-filtered2.jpg', filtered2)
```

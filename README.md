# pykuwahara

[![CI](https://github.com/yoch/pykuwahara/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yoch/pykuwahara/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yoch/pykuwahara/graph/badge.svg)](https://codecov.io/gh/yoch/pykuwahara)

Kuwahara filter in Python (NumPy + OpenCV).

> The Kuwahara filter is a non-linear smoothing filter used in image processing for adaptive noise reduction. It is able to apply smoothing on the image while preserving the edges.
> Source: [Wikipedia](https://en.wikipedia.org/wiki/Kuwahara_filter)

Two variants: uniform window mean, or Gaussian window mean ([ImageMagick-style notes](http://www.fmwconcepts.com/imagemagick/kuwahara/index.php)).

## Install

Python **3.9+**.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install pykuwahara
```

Dev / tests:

```bash
pip install -e ".[dev]"
pytest -q
pytest -q --cov=pykuwahara --cov-report=term-missing
```

Dependencies: `numpy`, `opencv-python-headless`. Do not install another PyPI package that ships its own `cv2` alongside it.

## Usage

### Basic example

```
import cv2
from pykuwahara import kuwahara

image = cv2.imread('lena_std.jpg')

filt1 = kuwahara(image, method='mean', radius=3)
filt2 = kuwahara(image, method='gaussian', radius=3)   # default sigma: OpenCV

cv2.imwrite('lena-kfilt-mean.jpg', filt1)
cv2.imwrite('lena-kfilt-gaus.jpg', filt2)
```

#### Original image
![Original image](/examples/lena_std.jpg)
#### Filtered with Kuwahara (mean)
![Mean method](/examples/lena-kfilt-mean.jpg)
#### Filtered with Kuwahara (gaussian)
![Gaussian method](/examples/lena-kfilt-gaus.jpg)


### Painting-style effect

```
import cv2
from pykuwahara import kuwahara

image = cv2.imread('photo.jpg')

filt1 = kuwahara(image, method='mean', radius=4)
filt2 = kuwahara(image, method='gaussian', radius=4, sigma=1.5)

cv2.imwrite('photo-kfilt-mean.jpg', filt1)
cv2.imwrite('photo-kfilt-gaus.jpg', filt2)
```

#### Original image (source: [wikipedia](https://en.wikipedia.org/wiki/File:Kuwahara_creates_artistic_photo.jpg))
![Original image](/examples/photo.jpg)
#### Filtered with Kuwahara (mean)
![Mean method](/examples/photo-kfilt-mean.jpg)
#### Filtered with Kuwahara (gaussian)
![Gaussian method](/examples/photo-kfilt-gaus.jpg)


### Colour: variance channel

For colour images, variance uses one channel (grayscale by default, or `image_2d`, or `grayconv`).

```
import cv2
from pykuwahara import kuwahara

image = cv2.imread('selfie.jpg')
image = (image / 255).astype('float32')

lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
l, a, b = cv2.split(lab_image)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)

filt1 = kuwahara(image, method='gaussian', radius=5, sigma=2., image_2d=l)
filt2 = kuwahara(image, method='gaussian', radius=5, sigma=2., image_2d=v)

cv2.imwrite('selfie-kfilt-gaus1.jpg', filt1 * 255)
cv2.imwrite('selfie-kfilt-gaus2.jpg', filt2 * 255)
```

#### Original image ([source](https://stackoverflow.com/questions/47017741/image-filter-to-cartoonize-and-colorize#47087756))
![Original image](/examples/selfie.jpg)
#### Filtered with Kuwahara on L (Lab)
![Lab](/examples/selfie-kfilt-gaus1.jpg)
#### Filtered with Kuwahara on V (HSV)
![HSV](/examples/selfie-kfilt-gaus2.jpg)

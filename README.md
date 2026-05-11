# pykuwahara

Kuwahara filter in Python (numpy + OpenCV).

> The Kuwahara filter is a non-linear smoothing filter used in image processing for adaptive noise reduction. It is able to apply smoothing on the image while preserving the edges.
> Source: [Wikipedia](https://en.wikipedia.org/wiki/Kuwahara_filter)

This implementation provide two variants of the filter:
- The classic one, using a uniform kernel to compute the window mean.
- A gaussian based filter, by computing the window gaussian mean. This is inspired by the [ImageMagick](http://www.fmwconcepts.com/imagemagick/kuwahara/index.php) approach.

## Installation

Requires **Python 3.9+**.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install pykuwahara
```

For development or running tests:

```bash
pip install -e ".[dev]"
pytest -q
```

### Dependencies

Runtime wheels pulled in by `pip`:

| Package | Role |
|--------|------|
| **numpy** | Arrays (`>=1.26,<3`). |
| [**opencv-python-headless**](https://pypi.org/project/opencv-python-headless/) | OpenCV **without GUI** (no HighGUI). Same main `cv2` API used here (`imgproc`); **no contrib modules**. Smaller and suited to servers and CI than `opencv-python`. Pip selects a wheel compatible with your NumPy line (for example NumPy 1.x pairs with OpenCV 4.11.x; NumPy 2.x uses newer wheels). |

Install **only one** of the PyPI meta-packages that ship `cv2` (`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`, `opencv-contrib-python-headless`). They replace each other; mixing them causes broken or unpredictable installs.

**Breaking change:** releases now depend on **`opencv-python-headless`** instead of `opencv-contrib-python`. This project only needs the standard `cv2` API. If you still have contrib installed, uninstall it and reinstall `pykuwahara` so pip resolves **`opencv-python-headless`**:

```bash
pip uninstall -y opencv-contrib-python opencv-contrib-python-headless opencv-python opencv-python-headless  # remove any old cv2 wheel
pip install pykuwahara
```

### PEP 668 (externally managed environments)

On recent Debian/Ubuntu (and similar), the system Python may refuse global `pip install` with an *externally-managed-environment* error. Use a **virtual environment** or a tool like **[uv](https://github.com/astral-sh/uv)**. Avoid `--break-system-packages` unless you fully accept overwriting distro Python.

### Test matrix (CI)

| Python | NumPy |
|--------|--------|
| 3.9 | 1.x (`>=1.26,<2`) |
| 3.11 | 1.x (`>=1.26,<2`) |
| 3.13 | 2.x (`>=2,<3`) |

\* CI installs NumPy and OpenCV in one step so pip resolves a compatible pair.

OpenCV is always **`opencv-python-headless`** from package metadata (not a separate CI axis).

## Usage

### Simple example

```
import cv2
from pykuwahara import kuwahara

image = cv2.imread('lena_std.jpg')

filt1 = kuwahara(image, method='mean', radius=3)
filt2 = kuwahara(image, method='gaussian', radius=3)    # default sigma: computed by OpenCV

cv2.imwrite('lena-kfilt-mean.jpg', filt1)
cv2.imwrite('lena-kfilt-gaus.jpg', filt2)
```

#### Original image
![Original image](/examples/lena_std.jpg)
#### Filtered with Kuwahara (mean)
![Mean method](/examples/lena-kfilt-mean.jpg)
#### Filtered with Kuwahara (gaussian)
![Gaussian method](/examples/lena-kfilt-gaus.jpg)


### Painting effect

Kuwahara filter can be used to apply a painting effet on pictures.

```
import cv2
from pykuwahara import kuwahara

image = cv2.imread('photo.jpg')

# Set radius according to the image dimensions and the desired effect
filt1 = kuwahara(image, method='mean', radius=4)
# NOTE: with sigma >= radius, this is equivalent to using 'mean' method
# NOTE: with sigma << radius, the radius has no effect
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


### Advanced usage

Color image are supported by grayscaling the source image and using the gray channel to calculate the variance.
The user can provide another channel at his convenience, and alternatively give the right color conversion code (default is `COLOR_BGR2GARY`).

```
import cv2
from pykuwahara import kuwahara

image = cv2.imread('selfie.jpg')
image = (image / 255).astype('float32')     # pykuwahara supports float32 as well

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
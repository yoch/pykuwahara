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

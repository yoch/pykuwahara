import cv2
from pykuwahara import kuwahara

image = cv2.imread('lena_std.jpg')

filt1 = kuwahara(image, method='mean', radius=5)
filt2 = kuwahara(image, method='gaussian', radius=5)    # default sigma: computed by OpenCV

cv2.imwrite('lena-kfilt-mean.jpg', filt1)
cv2.imwrite('lena-kfilt-gaus.jpg', filt2)


image = cv2.imread('lion.jpg')

filt1 = kuwahara(image, method='mean', radius=7)
filt2 = kuwahara(image, method='gaussian', radius=7)    # default sigma: computed by OpenCV

cv2.imwrite('lion-kfilt-mean.jpg', filt1)
cv2.imwrite('lion-kfilt-gaus.jpg', filt2)

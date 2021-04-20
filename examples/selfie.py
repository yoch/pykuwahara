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
#filt = kuwahara(lab_image, method='gaussian', radius=5, sigma=2., image_2d=l)

cv2.imwrite('selfie-kfilt-gaus1.jpg', filt1 * 255)
cv2.imwrite('selfie-kfilt-gaus2.jpg', filt2 * 255)
#cv2.imwrite('selfie-kfilt-gaus.jpg', cv2.cvtColor(filt, cv2.COLOR_Lab2BGR) * 255)

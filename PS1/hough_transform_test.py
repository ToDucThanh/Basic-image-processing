import cv2
from hough_lines_acc import *
from hough_peaks import *
from auto_canny import *

'''
img = cv2.imread('input/triangle.jpg', cv2.IMREAD_GRAYSCALE)

edge = auto_canny(img)
H, thetas, rhos = hough_lines_acc_fast(edge)
peaks = hough_peaks(H, num_peaks=10)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for peak in peaks:
    rho = rhos[peak[0]]
    theta = thetas[peak[1]] * np.pi / 180.0
    a = np.cos(theta)
    b = np.sin(theta)
    pt0 = rho * np.array([a, b])
    pt1 = tuple((pt0 + 1000 * np.array([-b, a])).astype(int)) # why do we have this -> i understood :)))
    pt2 = tuple((pt0 - 1000 * np.array([-b, a])).astype(int)) # why do we have this, what is 1000?
    cv2.line(img_color, pt1, pt2, (0,255,0), 2)


cv2.imshow('img', img_color)
cv2.waitKey()
'''


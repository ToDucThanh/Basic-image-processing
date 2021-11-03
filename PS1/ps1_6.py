import cv2

from auto_canny import *
from hough_peaks import *
from hough_lines_acc import hough_lines_acc_fast
from hough_lines_draw import *


orig = cv2.imread('input/ps1-input2.png', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(orig, (21, 21), 2)
edge = auto_canny(blur)
H, thetas, rhos = hough_lines_acc_fast(edge)
peaks = hough_peaks(H, num_peaks=10, nhoodsize=25, threshold=150)
color_img = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
line = hough_lines_draw(color_img, peaks, rhos, thetas)
cv2.imshow('Line image', line)
cv2.waitKey()
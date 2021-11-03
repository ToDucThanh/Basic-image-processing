import numpy as np
import cv2
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from auto_canny import *
import time


def ps1_2():
    start = time.time()
    img = cv2.imread('input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
    edge = auto_canny(img)
    H, thetas, rhos = hough_lines_acc_fast(edge)
    peaks = hough_peaks(H, num_peaks=10)
    H1 = H.copy()
    for peak in peaks:
        cv2.circle(H1, tuple(peak[::-1]), 4, (255, 255, 255), -1)
    #cv2.imshow('img', H1)
    #cv2.waitKey()
    col_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_draw = hough_lines_draw(col_img, peaks, rhos, thetas)
    #cv2.imwrite('output/ps1-2-final.png', img_draw)
    print('time elapsed %f' %(time.time() - start))
    cv2.imshow('img', img_draw)
    cv2.waitKey()


ps1_2()



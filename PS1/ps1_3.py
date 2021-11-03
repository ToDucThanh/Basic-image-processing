import cv2
import numpy as np
import time
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from auto_canny import *

def ps1_3():
    start_time = time.time()
    noisy_img = cv2.imread('input/ps1-input0-noise.png', cv2.IMREAD_GRAYSCALE)
    #  3a: smooth the noisy image using gaussian blurring
    smoothed_img = cv2.GaussianBlur(noisy_img, (23,) * 2, 4.5)
    #  3b: perform edge detection on both images using Canny
    min_val = 20
    max_val = 2 * min_val
    noisy_edge_img = cv2.Canny(noisy_img, min_val, max_val)
    edge_img = cv2.Canny(smoothed_img, min_val, max_val)
    #  3c: apply hough line detection to the smoothed image
    H, thetas, rhos = hough_lines_acc(edge_img)
    peaks = hough_peaks(H, num_peaks=20, threshold=50)
    for peak in peaks:
        cv2.circle(H, tuple(peak[::-1]), 5, (255, 255, 255), -1)
    # indices, H1 = hough_peaks_2(H, num_peaks=10, nhood_size=10)
    color_img = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR)
    img_draw = hough_lines_draw(color_img, peaks, rhos, thetas)
    #  save the produced images
    '''
    cv2.imwrite('output/ps1-3-a-1.png', smoothed_img)
    cv2.imwrite('output/ps1-3-b-1.png', noisy_edge_img)
    cv2.imwrite('output/ps1-3-b-2.png', edge_img)
    cv2.imwrite('output/ps1-3-c-1.png', H)
    cv2.imwrite('output/ps1-3-c-2.png', img_draw)
    '''
    print('3) Time elapsed: %.2f s' % (time.time() - start_time))
    cv2.imshow('Hough space', H)
    cv2.imshow('Line segments on image', img_draw)
    cv2.waitKey()

ps1_3()

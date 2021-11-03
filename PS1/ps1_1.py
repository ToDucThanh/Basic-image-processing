import cv2
import numpy as np
from auto_canny import *
import time

input_path = 'input/'
output_path = 'output/'


def ps1_a():
    start = time.time()
    img = cv2.imread(input_path + 'ps1-input0.jpg', cv2.IMREAD_GRAYSCALE)
    edge_img = auto_canny(img, 0.5)

    cv2.imwrite(output_path + 'ps1-input0-a.png', edge_img)
    print('Time elapsed: %f' % (time.time() - start))


#ps1_a()

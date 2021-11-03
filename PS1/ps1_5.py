import cv2

from hough_circles_acc import *
from find_circles import *
from hough_circles_drawn import *


# Load an edge image
edge = cv2.imread('output/ps1-4-b-1.png', cv2.IMREAD_GRAYSCALE)
# Detect circles
centers, radius = find_circles(edge, (20, 50), nhoodsize=50, threshold=130, verbose=True)
# Draw circles on the original image
img = cv2.imread('input/ps1-input1.png')
img_draw = img.copy()
hough_circles_drawn(img_draw, centers, radius)
#cv2.imwrite('output/ps1-5.png', img_draw)

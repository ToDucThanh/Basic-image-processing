from auto_canny import *
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
img = cv2.imread('input/ps1-input1.png', cv2.IMREAD_GRAYSCALE)
blur_img = cv2.GaussianBlur(img, (17, 17), 4)
edge_with_filter = auto_canny(blur_img)
H, thetas, rhos = hough_lines_acc_fast(edge_with_filter, rho_res=2)
peaks = hough_peaks(H, num_peaks=10)
for peak in peaks:
    cv2.circle(H, tuple(peak[::-1]), 3, (255, 255, 255), -1)
#color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#line = hough_lines_draw(color_img, peaks, rhos, thetas)
"""
cv2.imwrite('output/ps1-4-a-1.png', blur_img)
cv2.imwrite('output/ps1-4-b-1.png', edge_with_filter)
cv2.imwrite('output/ps1-4-c-1.png', H)
cv2.imwrite('output/ps1-4-c-2.png', line)
"""
cv2.imwrite('output/test2.png', H)

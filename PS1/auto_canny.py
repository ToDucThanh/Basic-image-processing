import numpy as np
import cv2


def auto_canny(image, const=0.33):
    m = np.median(image)
    lower = int(max(0, (1 - const) * m))
    upper = int(min(255, (1 + const) * m))
    edge = cv2.Canny(image, lower, upper)
    return edge



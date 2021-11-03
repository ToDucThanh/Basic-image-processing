import cv2


def hough_circles_drawn(img, centers, radius):
    for subcenters, radii in zip(centers, radius):
        for center in subcenters:
            cv2.circle(img, radius=radii, center=tuple(center[::-1]), color=(0, 255, 0))



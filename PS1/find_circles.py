
from hough_circles_acc import *
from hough_peaks import *


def find_circles(edge_img, radius_range=(1, 2), threshold=100, nhoodsize=5, num_peaks=10, verbose=False):
    n = radius_range[1] - radius_range[0]
    H_size = (n,) + edge_img.shape
    H = np.zeros(H_size)
    centers = ()
    radius = np.arange(radius_range[0], radius_range[1])
    valid_radius = np.array([], dtype=np.uint)
    num_circles = 0
    for i in range(len(radius)):
        H[i] = hough_circles_acc(edge_img, radius[i])
        peaks = hough_peaks(H[i], num_peaks=num_peaks, threshold=threshold, nhoodsize=nhoodsize)
        if peaks.shape[0]:
            valid_radius = np.append(valid_radius, radius[i])
            centers = centers + (peaks,)
            for peak in peaks:
                cv2.circle(edge_img, tuple(peak[::-1]), radius[i]+1, (0, 0, 0), -1) # this step is to cancel the
                # circles found
        num_circles += peaks.shape[0]
        if verbose:
            print('Progress: %d%% - Circles detected: %d' % (100*i/len(radius), num_circles))
    if verbose:
        print('Total circles detected: %d' % num_circles)
    centers = np.array(centers, dtype=object)
    return centers, valid_radius.astype(np.uint)


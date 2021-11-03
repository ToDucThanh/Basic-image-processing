import cv2
import numpy as np


def hough_lines_acc(img, rho_res=1, thetas=np.arange(-90, 90, 1)):
    num_edge_points = (img == 255).sum()
    i, j = np.where(img == 255)  # i is y and j is x
    # print(i.shape, j.shape)
    # print('Time elapsed: %f' %(time.time() - start))
    diagonal = np.ceil(np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2))
    rhos = np.arange(-diagonal, diagonal + 1, rho_res)
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    thetas -= min(min(thetas), 0)
    # edge_indices = np.vstack((i, j))
    # print(edge_indices.shape)

    for k in range(num_edge_points):
        x = j[k]
        y = i[k]
        for theta in thetas:
            rho = x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))
            rho = int(rho + diagonal)
            H[rho, theta] += 1
    dst_img = np.zeros_like((100, 100), dtype=np.uint8)
    normalize = cv2.normalize(H, dst_img, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return normalize, thetas, rhos

# cv2.imshow('img', normalize)
# cv2.waitKey()


def hough_lines_acc_fast(img, rho_res=1, thetas=np.arange(-90,90,1)):
    rho_max = int(np.linalg.norm(img.shape-np.array([1,1]), 2))
    rhos = np.arange(-rho_max, rho_max, rho_res)
    thetas -= min(min(thetas),0)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    yis, xis = np.nonzero(img) # use only edge points
    for idx in range(len(xis)):
        x = xis[idx]
        y = yis[idx]
        temp_rhos = x * np.cos(np.deg2rad(thetas)) + y * np.sin(np.deg2rad(thetas))
        temp_rhos = temp_rhos / rho_res + rho_max
        m, n = accumulator.shape
        valid_idxs = np.nonzero((temp_rhos < m) & (thetas < n))
        temp_rhos = temp_rhos[valid_idxs]
        temp_thetas = thetas[valid_idxs]
        c = np.stack([temp_rhos,temp_thetas], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1]))) # what this line mean???
        _,idxs,counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[idxs].astype(np.uint)
        accumulator[uc[:,0], uc[:,1]] += counts.astype(np.uint)
    accumulator = cv2.normalize(accumulator, accumulator, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return accumulator, thetas, rhos


def hough_lines_acc_2(bw, rho_step=1, theta=np.linspace(-90, 89, 180)):
    max_rho = np.linalg.norm(bw.shape)
    rho = np.arange(-max_rho, max_rho, step=rho_step)
    h = np.zeros((rho.size, theta.size))
    for (y, x), is_edge in np.ndenumerate(bw):
        if is_edge:
            for t_i, t in np.ndenumerate(theta):
                r = x * np.cos(np.deg2rad(t)) + y * np.sin(np.deg2rad(t))
                nearest_r_i = np.abs(rho - r).argmin()
                h[nearest_r_i, t_i] += 1

    accumulator = cv2.normalize(h, h, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return accumulator, theta, rho

"""
bw = cv2.imread('output/ps1-input0-a.png', cv2.IMREAD_GRAYSCALE)
H, thetas, rhos = hough_lines_acc_fast(bw, rho_res=1)
cv2.imwrite('output/ps1-2-a-1.png', H)
"""


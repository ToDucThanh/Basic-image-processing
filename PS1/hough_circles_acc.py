import numpy as np


def hough_circles_acc(edge, radius):
    y_edge, x_edge = np.nonzero(edge)
    H = np.zeros(edge.shape)
    m, n = edge.shape
    theta = np.arange(0, 360)
    for i in range(len(x_edge)):
        x = x_edge[i]
        y = y_edge[i]
        a = np.ceil(y + radius * np.sin(np.deg2rad(theta)))
        b = np.ceil((x - radius * np.cos(np.deg2rad(theta))))
        valid_idx = np.nonzero((a < m) & (b < n) & (a >= 0) & (b >= 0))
        a, b = a[valid_idx], b[valid_idx]
        c = np.stack([a, b], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
        _, idx, counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[idx].astype(np.uint)
        H[uc[:, 0], uc[:, 1]] += counts.astype(np.uint)
    return H


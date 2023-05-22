import numpy as np


def compute_img_idx_to_track_idx(t_track, t_image):
    x, counts = np.unique(t_track, return_counts=True)
    i, j = (x.reshape((-1,1)) == t_image.reshape((1,-1))).nonzero()
    deltas = np.zeros_like(t_image)

    deltas[j] = counts[i]

    idx = np.concatenate([np.array([0]), deltas]).cumsum()
    return np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")
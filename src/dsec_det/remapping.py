import numpy as np


def conf_to_K(conf):
    K = np.eye(3)
    K[[0, 1, 0, 1], [0, 1, 2, 2]] = conf
    return K

def compute_remapping(calibration, mapping):
    mapping = mapping['rectify_map']

    K_r0 = conf_to_K(calibration['intrinsics']['camRect0']['camera_matrix'])
    K_r1 = conf_to_K(calibration['intrinsics']['camRect1']['camera_matrix'])

    R_r0_0 = np.array(calibration['extrinsics']['R_rect0'])
    R_r1_1 = np.array(calibration['extrinsics']['R_rect1'])
    R_1_0 = np.array(calibration['extrinsics']['T_10'])[:3, :3]

    # read from right to left:
    # rect. cam. 1 -> norm. rect. cam. 1 -> norm. cam. 1 -> norm. cam. 0 -> norm. rect. cam. 0 -> rect. cam. 0
    P_r0_r1 = K_r0 @ R_r0_0 @ R_1_0.T @ R_r1_1.T @ np.linalg.inv(K_r1)

    H, W = mapping.shape[:2]
    coords_hom = np.concatenate((mapping, np.ones((H, W, 1))), axis=-1)
    mapping = (np.linalg.inv(P_r0_r1) @ coords_hom[..., None]).squeeze()
    mapping = mapping[...,:2] / mapping[..., -1:]
    mapping = mapping.astype('float32')

    return mapping

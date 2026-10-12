import numpy as np
import os
import os.path as osp
import imageio
from tqdm import tqdm
import cv2



from .data import load_middlebury_data


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max

def compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    i_R_w, j_R_w : [3,3]
    i_T_w, j_T_w : [3,1]
        p_i = i_R_w @ p_w + i_T_w
        p_j = j_R_w @ p_w + j_T_w

    Returns
    -------
    i_R_j, i_T_j: [3,3], [3,1],
        p_i = i_R_j @ p_j + i_T_j
    B: float,
        the baseline
    """

    #====== STUDENT CODE STARTS ======#
    i_R_j = i_R_w @ np.linalg.inv(j_R_w)
    i_T_j = i_T_w - i_R_j @ j_T_w
    B = np.linalg.norm(i_T_j)
    #====== STUDENT CODE ENDS   ======#

    return i_R_j, i_T_j, B

def rectify_2view(rgb_i, rgb_j, rect_R_i, rect_R_j, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    rect_R_i,rect_R_j : [3,3]
        p_rect_left = rect_R_i @ p_i
        p_rect_right = rect_R_j @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    rgb_i_rect, rgb_j_rect: [H,W,3],[H,W,3],
        the rectified images.
    K_i_corr, K_j_corr: [3,3],[3,3],
        the corrected camera projection matrix.

    NOTE: K_i_corr, K_j_corr have been computed for you! YOU DON'T NEED TO CHANGE THIS.
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ rect_R_i @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ rect_R_j @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    #====== STUDENT CODE STARTS ======#
    H_i = K_i_corr @ rect_R_i @ np.linalg.inv(K_i)
    H_j = K_j_corr @ rect_R_j @ np.linalg.inv(K_j)

    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, (w_max, h_max))
    #====== STUDENT CODE ENDS   ======#

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_rectification_R(i_T_j):
    """Compute the rectification Rotation

    Parameters
    ----------
    i_T_j : [3,1]

    Returns
    -------
    p_rect: [3,3],
        p_rect = rect_R_i @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    EPS = 1e-8
    e_i = i_T_j.squeeze(-1) / (i_T_j.squeeze(-1)[1] + EPS)

    #====== STUDENT CODE STARTS ======#
    r2 = e_i / (np.linalg.norm(e_i) + EPS)
    r1 = np.array([e_i[1], -e_i[0], 0.0], dtype=np.float64)
    r1 = r1 / (np.linalg.norm(r1) + EPS)
    r3 = np.cross(r1, r2)
    r3 = r3 / (np.linalg.norm(r3) + EPS)
    rect_R_i = np.stack([r1, r2, r3], axis=0)
    #====== STUDENT CODE ENDS   ======#

    return rect_R_i

def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    ssd: [M,N],
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    #====== STUDENT CODE STARTS ======#
    M = src.shape[0]
    N = dst.shape[0]

    diff = src[:, None, :, :] - dst[None, :, :, :]
    ssd = np.sum(diff * diff, axis=(2, 3))
    #====== STUDENT CODE ENDS   ======#

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    sad: [M,N],
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    #====== STUDENT CODE STARTS ======#
    M = src.shape[0]
    N = dst.shape[0]

    sad = np.sum(np.abs(src[:, None, :, :] - dst[None, :, :, :]), axis=(2, 3))
    #====== STUDENT CODE ENDS   ======#

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    zncc: [M,N],
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    #====== STUDENT CODE STARTS ======#
    EPS = 1e-8
    M = src.shape[0]
    N = dst.shape[0]

    zncc = np.zeros((M, N), dtype=np.float64)
    for c in range(src.shape[-1]):
        src_c = src[:, :, c]
        dst_c = dst[:, :, c]

        src_mean = np.mean(src_c, axis=1, keepdims=True)
        dst_mean = np.mean(dst_c, axis=1, keepdims=True)
        src_std = np.std(src_c, axis=1, keepdims=True)
        dst_std = np.std(dst_c, axis=1, keepdims=True)

        numerator = np.sum((src_c - src_mean)[:, None, :] * (dst_c - dst_mean)[None, :, :], axis=2)
        denominator = src_std * dst_std.T

        channel_zncc = np.zeros((M, N), dtype=np.float64)
        valid = denominator > EPS
        channel_zncc[valid] = numerator[valid] / denominator[valid]
        zncc += channel_zncc
    #====== STUDENT CODE ENDS   ======#

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    patch_buffer: [H, W, k_size**2, 3],
        The patch buffer for each pixel
    """

    #====== STUDENT CODE STARTS ======#
    assert k_size % 2 == 1, "k_size must be odd"

    h, w = image.shape[:2]
    pad = k_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="constant")

    patch_buffer = np.zeros((h, w, k_size * k_size, 3), dtype=image.dtype)
    idx = 0
    for dv in range(k_size):
        for du in range(k_size):
            patch_buffer[:, :, idx, :] = padded[dv : dv + h, du : du + w, :]
            idx += 1
    #====== STUDENT CODE ENDS   ======#

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel,  img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func: function, optional
        the function used to compute the patch buffer, by default image2patch
        (there is NO NEED to alter this argument)

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    #====== STUDENT CODE STARTS ======#
    disp_map = np.zeros(rgb_i.shape[:2], dtype=np.float64)
    lr_consistency_mask = np.ones_like(disp_map, dtype=np.float64)

    rgb_i = rgb_i.astype(np.float64) / 255.0
    rgb_j = rgb_j.astype(np.float64) / 255.0

    # 1. Compute the patch buffers for both images
    # NOTE: when computing patches, please use the syntax:
    # patch_buffer = img2patch_func(image, k_size)
    # DO NOT DIRECTLY USE: patch_buffer = image2patch(image, k_size), as it may cause errors in the autograder
    patches_i = img2patch_func(rgb_i, k_size)
    patches_j = img2patch_func(rgb_j, k_size)

    # 3. Compute disparity candidates
    h, w = rgb_i.shape[:2]
    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    valid_disp_mask = disp_candidates > 0.0

    # 2. Compute the disparity and LR consistency for each pixel
    for u in range(w):
        buf_i = patches_i[:, u]
        buf_j = patches_j[:, u]
        value = kernel_func(buf_i, buf_j)

        upper = value.max() + 1.0
        value[~valid_disp_mask] = upper

        best_right = np.argmin(value, axis=1)
        disp_map[:, u] = disp_candidates[np.arange(h), best_right]

        best_left = np.argmin(value, axis=0)
        lr_consistency_mask[:, u] = (best_left[best_right] == np.arange(h)).astype(np.float64)
    #====== STUDENT CODE ENDS   ======#

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    dep_map: [H,W]
        depth map
    xyz_cam: [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame.
        You may find using np.meshgrid helpful here.
    """

    #====== STUDENT CODE STARTS ======#
    fy = K[1, 1]
    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    dep_map = np.zeros_like(disp_map, dtype=np.float64)
    valid = disp_map > 1e-8
    dep_map[valid] = fy * B / disp_map[valid]

    h, w = disp_map.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    x = (uu - cx) * dep_map / fx
    y = (vv - cy) * dep_map / fy
    z = dep_map
    xyz_cam = np.stack([x, y, z], axis=-1)
    #====== STUDENT CODE ENDS   ======#

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    c_R_w,
    c_T_w,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is:
    given pcl_cam [N,3], c_R_w [3,3] and c_T_w [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate

    NOTE: This can be done with one line of code!
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    # Statistical filtering from the course framework, with a lightweight SciPy backend.
    from scipy.spatial import cKDTree
    if len(pcl_cam) > 10:
        distances, _ = cKDTree(pcl_cam).query(pcl_cam, k=11)
        mean_distance = distances[:, 1:].mean(axis=1)
        keep = mean_distance <= mean_distance.mean() + 2.0 * mean_distance.std()
        active = np.flatnonzero(mask.reshape(-1) > 0)
        mask.reshape(-1)[active[~keep]] = 0

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    #====== STUDENT CODE STARTS ======#
    pcl_world = (c_R_w.T @ (pcl_cam.T - c_T_w)).T
    #====== STUDENT CODE ENDS   ======#

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    i_R_w, i_T_w = view_i["R"], view_i["T"][:, None]  # p_i = i_R_w @ p_w + i_T_w
    j_R_w, j_T_w = view_j["R"], view_j["T"][:, None]  # p_j = j_R_w @ p_w + j_T_w

    i_R_j, i_T_j, B = compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w)
    assert i_T_j[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    rect_R_i = compute_rectification_R(i_T_j)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        rect_R_i,
        rect_R_i @ i_R_j,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        rect_R_i @ i_R_w,
        rect_R_i @ i_T_w,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()

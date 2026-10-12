import numpy as np

from vision5800.stereo import compute_right2left_transformation, compute_dep_and_pcl, compute_disparity_map, ssd_kernel


def test_relative_transform_maps_world_point_consistently():
    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    ti, tj = np.array([[0.1], [0.4], [0.2]]), np.array([[0.2], [-0.1], [0.3]])
    relative_r, relative_t, baseline = compute_right2left_transformation(np.eye(3), ti, rotation, tj)
    world = np.array([[0.5], [0.2], [1.0]])
    np.testing.assert_allclose(relative_r @ (rotation @ world + tj) + relative_t, world + ti)
    np.testing.assert_allclose(baseline, np.linalg.norm(relative_t))


def test_depth_reprojection_and_invalid_disparity():
    k = np.array([[100., 0, 1], [0, 100., 1], [0, 0, 1]])
    disparity = np.full((3, 3), 10.)
    disparity[0, 0] = 0
    depth, points = compute_dep_and_pcl(disparity, 0.2, k)
    np.testing.assert_allclose(points[1, 1], [0, 0, 2])
    assert depth[0, 0] == 0
    projected = (k @ points[1:, 1:].reshape(-1, 3).T).T
    projected = projected[:, :2] / projected[:, 2:]
    np.testing.assert_allclose(projected, [[1, 1], [2, 1], [1, 2], [2, 2]])


def test_matching_recovers_known_vertical_shift():
    rng = np.random.default_rng(42)
    left = rng.integers(0, 255, (10, 3, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:-2] = left[2:]
    disparity, mask = compute_disparity_map(left, right, 0, k_size=1, kernel_func=ssd_kernel)
    np.testing.assert_allclose(disparity[2:], 2)
    np.testing.assert_allclose(mask[2:], 1)

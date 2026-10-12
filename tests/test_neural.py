import pytest
torch = pytest.importorskip('torch')
from vision5800.nerf import get_rays, volumetric_rendering, nerf_model
from vision5800.neural_experiment import render_rays


def test_camera_rays_follow_pose_and_principal_point():
    k = torch.tensor([[2., 0, 1], [0, 2., 1], [0, 0, 1]])
    center = torch.tensor([1., 2., 3.])
    origins, directions = get_rays(3, 3, k, torch.eye(3), center)
    torch.testing.assert_close(origins[0, 0], center)
    torch.testing.assert_close(directions[1, 1], torch.tensor([0., 0., 1.]))


def test_empty_and_opaque_volume_and_gradients():
    rgb = torch.full((1, 1, 2, 3), 0.4, requires_grad=True)
    depths = torch.tensor([[[0., 1.]]])
    empty = volumetric_rendering(rgb, torch.zeros(1, 1, 2), depths)
    torch.testing.assert_close(empty, torch.zeros(1, 1, 3))
    opaque = volumetric_rendering(rgb, torch.ones(1, 1, 2), depths)
    torch.testing.assert_close(opaque, torch.full((1, 1, 3), 0.4))
    opaque.sum().backward()
    assert torch.isfinite(rgb.grad).all()


def test_ray_chunks_preserve_rendered_result():
    torch.manual_seed(4)
    model = nerf_model(32, 2, 1)
    origins = torch.zeros(5, 3)
    directions = torch.tensor([[0., 0., 1.]]).expand(5, 3)
    whole = render_rays(model, origins, directions, 4, 2, 1)
    chunks = torch.cat([render_rays(model, origins[:2], directions[:2], 4, 2, 1),
                        render_rays(model, origins[2:], directions[2:], 4, 2, 1)])
    torch.testing.assert_close(whole, chunks)

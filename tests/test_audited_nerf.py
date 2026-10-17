import numpy as np
import torch
from vision5800.nerf import get_rays,stratified_sampling,volumetric_rendering
from vision5800.research import quality


def test_image_down_axis_and_world_pose_projection():
    k=torch.tensor([[2.,0,1],[0,2.,1],[0,0,1]])
    rotation=torch.tensor([[0.,0,1],[0,1,0],[-1,0,0]])
    center=torch.tensor([1.,2.,3.])
    origins,directions=get_rays(3,3,k,rotation,center)
    point=origins[2,1]+3*directions[2,1]
    camera=rotation.T@(point-center)
    projected=k@camera
    torch.testing.assert_close(projected[:2]/projected[2],torch.tensor([1.,2.]))
    assert camera[1]>0


def test_finite_constant_medium_is_analytic_and_gradient_survives():
    origin=torch.zeros(1,1,3);direction=torch.tensor([[[0.,0,2.]]])
    _,depths=stratified_sampling(origin,direction,0.,1.,8)
    # softplus(0)=ln2; world path length=2, opacity=1-exp(-2*ln2)=.75
    raw=torch.zeros(1,1,8,requires_grad=True)
    rendered=volumetric_rendering(torch.ones(1,1,8,3),raw,depths,direction,1/8)
    torch.testing.assert_close(rendered,torch.full((1,1,3),.75))
    rendered.sum().backward();assert (raw.grad>0).all()


def test_samples_cover_interval_midpoints_and_ssim_identity():
    _,depths=stratified_sampling(torch.zeros(1,1,3),torch.ones(1,1,3),0.,2.,2)
    torch.testing.assert_close(depths,torch.tensor([[[.5,1.5]]]))
    x=np.random.default_rng(1).random((24,24,3))
    assert abs(quality(x,x)['ssim']-1)<1e-12


def test_ssim_constant_offset_has_known_value():
    a=np.full((24,24,3),.5);b=np.full((24,24,3),.6)
    expected=(2*.5*.6+.01**2)/(.5**2+.6**2+.01**2)
    assert abs(quality(a,b)['ssim']-expected)<1e-10

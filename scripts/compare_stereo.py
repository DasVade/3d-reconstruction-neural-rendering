"""Matched stereo configurations; report coverage/consistency, never depth accuracy."""
from pathlib import Path
import json
import time
import numpy as np
import cv2
from vision5800 import stereo as s
from vision5800.data import load_middlebury_data
from vision5800.experiments import resize_view
from vision5800.artifacts import save_ply,save_cloud_preview,save_map,save_image

root=Path(__file__).resolve().parents[1]
out=root/'outputs/stereo_controlled';out.mkdir(parents=True,exist_ok=True)
data=[resize_view(v,.25) for v in load_middlebury_data(str(root/'data/stereo/templeRing'))]
pairs=[(0,3),(2,4),(13,15)];rows=[]
for i,j in pairs:
    a,b=data[i],data[j]
    r,t,baseline=s.compute_right2left_transformation(a['R'],a['T'][:,None],b['R'],b['T'][:,None])
    rect=s.compute_rectification_R(t)
    left,right,kl,kr=s.rectify_2view(a['rgb'],b['rgb'],rect,rect@r,a['K'],b['K'],20,20)
    for name,kernel in [('ssd',s.ssd_kernel),('sad',s.sad_kernel),('zncc',s.zncc_kernel)]:
        folder=out/f'{i}_{j}_{name}';folder.mkdir(exist_ok=True)
        times=[]
        for repeat in range(3):
            start=time.perf_counter()
            disparity,lr=s.compute_disparity_map(left,right,kr[1,2]-kl[1,2],5,kernel)
            depth,xyz=s.compute_dep_and_pcl(disparity,baseline,kl)
            mask,points,_,colors=s.postprocess(depth,left,xyz,rect@a['R'],rect@a['T'][:,None],
                                             consistency_mask=lr,z_near=.5,z_far=.6)
            times.append(time.perf_counter()-start)
        # Denominator includes all pixels on the common rectified canvas, including padding/background.
        valid=np.isfinite(depth)&(depth>0)
        row={'pair':[i,j],'kernel':name,'shape':list(depth.shape),'matching_filter_median_seconds':float(np.median(times)),
             'runtime_repeats_seconds':times,'positive_depth_fraction':float(valid.mean()),
             'lr_consistency_fraction':float((lr>0).mean()),'final_valid_depth_fraction':float((mask>0).mean()),'point_count':len(points)}
        rows.append(row);print(json.dumps(row),flush=True)
        save_ply(folder/'reconstruction.ply',points,colors);save_cloud_preview(folder/'pointcloud.png',points,colors)
        save_map(folder/'depth.png',depth);save_image(folder/'lr.png',lr);save_image(folder/'valid.png',mask>0)
        save_image(folder/'rectified.png',np.concatenate([left,right],axis=1))
        np.savez_compressed(folder/'maps.npz',depth=depth,disparity=disparity,lr=lr,final_mask=mask)
result={'settings':{'scale':.25,'patch_size':5,'depth_near':.5,'depth_far':.6,'hsv_threshold':45,
                    'morph_close_kernel':11,'outlier_neighbors':10,'outlier_std_ratio':2,
                    'rectification_padding':20,'pairs':pairs,'repeats':3,
                    'denominator':'all pixels in each pair rectified canvas; identical per kernel',
                    'timing':'matching + depth + filtering; excludes common rectification, file IO and plotting',
                    'accuracy':'No local depth ground truth; coverage and consistency are not accuracy.'},'rows':rows}
(out/'metrics.json').write_text(json.dumps(result,indent=2))

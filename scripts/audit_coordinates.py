"""Validate camera conventions against supplied poses and image correspondences."""
from pathlib import Path
import json
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
data = np.load(root / 'data/nerf/lego_data.npz')
poses, images, k = data['poses'], data['images'], data['intrinsics']
centers = poses[:, :3, 3]
def skew(t):
    x,y,z=t
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
sift = cv2.SIFT_create()
features=[]
for image in images:
    gray=cv2.cvtColor(np.uint8(np.clip(image,0,1)*255),cv2.COLOR_RGB2GRAY)
    features.append(sift.detectAndCompute(gray,None))
residuals={1:[], -1:[]}
pairs=[]
for i in range(0,90,6):
    distances=np.linalg.norm(centers-centers[i],axis=1)
    distances[i]=np.inf
    j=int(np.argmin(distances))
    a,da=features[i];b,db=features[j]
    if da is None or db is None:continue
    matches=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[m for m,n in matches if m.distance < .75*n.distance]
    if len(good)<3:continue
    x=np.array([[*a[m.queryIdx].pt,1] for m in good]);y=np.array([[*b[m.trainIdx].pt,1] for m in good])
    ri, rj=poses[i,:3,:3],poses[j,:3,:3]
    rel=rj.T@ri;t=rj.T@(centers[i]-centers[j]);e=skew(t)@rel
    for sign in (1,-1):
        s=np.diag([1,sign,1]);f=np.linalg.inv(k).T@s@e@s@np.linalg.inv(k)
        fx=x@f.T;fty=y@f
        error=np.abs(np.sum(y*fx,axis=1))/np.sqrt(np.sum(fx[:,:2]**2+fty[:,:2]**2,axis=1)+1e-20)
        residuals[sign].extend(error.tolist())
    pairs.append([i,j,len(good)])
result={'camera_count':len(poses),'image_shape':list(images.shape[1:]),
 'mean_forward_radial_cosine':float(np.mean(np.sum(poses[:,:3,2]*centers,axis=1)/np.linalg.norm(centers,axis=1))),
 'rotation_determinant_range':[float(np.linalg.det(poses[:,:3,:3]).min()),float(np.linalg.det(poses[:,:3,:3]).max())],
 'pairs':pairs,'sampson_distance_pixels':{str(s):{'matches':len(v),'median':float(np.median(v))} for s,v in residuals.items()},
 'definition':'K inverse pixels, camera direction (x, sign*y, +1); descriptors filtered by Lowe ratio .75, no geometry-based inlier filtering.'}
out=root/'docs/results/coordinate_audit.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

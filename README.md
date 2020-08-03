# TransparentShapeRealData
The real transparent shape data creation pipeline for [Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/), CVPR'20

## Pre-processed Real Data
We have our pre-processed read shape used in our paper. The download link is [here]().

## Process Your Own Captured Data

### Step 0: Capture Data and Get Segmentation Maps
First, you need to prepare a mirror ball and the transparent shape(s) you want to reconstruct. Select a scene and fix the mirror ball and the transparent shape(s) alternately at the center. Use your camera to capture
- at least 5 images (the more the better) per transparent shape and get segmentation maps for each shape image.
- several images for mirror ball (to compute full environment map and remove the photographer in the image) under different views around it. 

### Step 1: Obtain Camera Poses
Put the captured images (mirror ball and transparent shape) under a single folder, run Structure-from-Motion algorithm (here we use [COLMAP](https://colmap.github.io)) to obtain camera poses for those images.

### Step 2: Detect Mirror Ball Location
```
python ballDetect.py
```

### Step 3: Compute Environment Map
```
python computeEnvMap.py
```

### Step 4: Build Visual Hull
```
python computeVisualHull.py
```

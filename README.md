# TransparentShapeRealData
The real transparent shape data creation pipeline for [Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/), CVPR'20

## Pre-processed Real Data
We have our pre-processed read shape used in our paper. The download link is [here]().

## Process Your Own Captured Data

### Step 0: Capture Data and Get Segmentation Maps
First, you need to prepare a mirror ball and the transparent shape(s) you want to reconstruct. Select a scene and fix the mirror ball and the transparent shape(s) alternately at the center. Use your camera to capture
- several images for mirror ball (to compute full environment map and remove the photographer in the image) under different views around it. 
- at least 5 images (the more the better) per transparent shape and get segmentation maps for each shape image.

### Step 1: Obtain Camera Poses
Put the captured images (mirror ball and transparent shape) under a single folder, run Structure-from-Motion algorithm (here we use [COLMAP](https://colmap.github.io) automatic reconstruction with shared intrinsics and sparse model) to obtain camera poses for those images. Next we will assume the camera poses are stored in `$SceneRoot/sparse/0/images.txt`
- Note that the mirror ball images are assumed to be captured before shape images so that the camera poses of the mirror ball images will appear before the shapes. This assumption will be used in Step 2.

### Step 2: Detect Mirror Ball Location
Requirement: OpenCV (tested on v3.4.2)
Make a copy of mirror images in `$SceneRoot/MirrorBall_Imgs`. The script will use HoughCircles algorithm to detect the mirror ball locations and output circle locations in `$SceneRoot/ballLoc_init.txt`. Use the optional tag `--isPrecise` to manually check detection results.
```
python ballDetect.py --scene $SceneRoot (--isPrecise)
```

### Step 3: Compute Environment Map
```
python computeEnvMap.py
```

### Step 4: Build Visual Hull
```
python computeVisualHull.py
```

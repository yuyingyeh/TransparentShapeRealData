# TransparentShapeRealData
The real transparent shape data creation pipeline for [Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/), CVPR'20

## Pre-processed Real Data
We have pre-processed real transparent shape data used in our paper. The download link is [here]().

## Process Your Own Captured Data
Requirements: 
- Python packages: OpenCV (tested on v3.4.2), pillow (tested on v7.2.0), scikit-image (tested on v0.16.2), trimesh (tested on v3.7.14)
- Enable `xvfb-run`. 
- [Meshlab](https://www.meshlab.net)

### Step 0: Capture Data and Get Segmentation Maps
First, you need to prepare a mirror ball and the transparent shape(s) you want to reconstruct. Select a scene and fix the mirror ball and the transparent shape(s) alternately at the center. Use your camera to capture
- several images for mirror ball (to compute full environment map and remove the photographer in the image) under different views around it. 
- at least 5 images (the more the better) per transparent shape

We assume all the captured images (mirror ball and transparent shapes) are stored in `$SceneRoot/All_Imgs` (used in Step 1). And from this folder, make a copy of mirror images to `$SceneRoot/MirrorBall_Imgs` (used in Step 2) and a copy of shape(s) to `$SceneRoot/$ShapeName` (and `$SceneRoot/$ShapeName2` if more than 1 shape) (used in Step 4).

Then, get segmentation masks for each shape image and save the mask in the alpha channel. The file names should be the following format:
- Shape Images: `$A_$B_$imgId.jpg`
- Shape Masks: `$A_$B_$imgId.png`
where `$A` and `$B` can be any string. 

We assume these masks are stored in `$SceneRoot/$ShapeName_Mask_$ViewNum`, where `$ShapeName` is the name of the shape and `$ViewNum` is the number of the views which used to reconstruct.

### Step 1: Obtain Camera Poses
From all the captured images (mirror ball and transparent shape) under a single folder `$SceneRoot/All_Imgs`, run Structure-from-Motion algorithm (here we use [COLMAP](https://colmap.github.io) automatic reconstruction with shared intrinsics and sparse model) to obtain camera poses for those images. Next we will assume the camera poses are stored in `$SceneRoot/sparse/0/images.txt` and the camera file is stored in `$SceneRoot/sparse/0/cameras.txt`
- Note that the mirror ball images are assumed to be captured before shape images so that the camera poses of the mirror ball images will appear before the shapes. This assumption will be used in Step 2.

### Step 2: Detect Mirror Ball Location
From the mirror images in `$SceneRoot/MirrorBall_Imgs`, the script will use HoughCircles algorithm to detect the mirror ball locations and output circle locations in `$SceneRoot/ballLoc.txt`. Use the optional tag `--isPrecise` to manually check detection results.
```
python 1_ballDetect.py --scene $SceneRoot (--isPrecise)
```

### Step 3: Compute Environment Map
The script will output estimated environment map in `$SceneRoot/env.png`. If you find the estimation too noisy, you may need to go back to Step 2 to choose accurate circles or change the paramters using `--param 1` or manually tune the parameters in this script. Use the optional tag `--showStepByStep` to save intermediate results for debugging.

```
python 2_computeEnvMap.py --scene $SceneRoot
```

### Step 4: Build Visual Hull
First put the original shape images in `$SceneRoot/$ShapeName` and mask images in `$SceneRoot/$ShapeName_Mask_$ViewNum`. The script will run visual hull algorithm and then do subdivision with meshlab to create a visual hull initialized mesh.
```
python 3_computeVisualHull.py --scene $SceneRoot --shapeName $ShapeName --nViews $ViewNum
```
The script will output visual hull mesh `$ShapeName_visualHull_$ViewNum.ply` and its subdivised mesh `$ShapeName_visualHullSubd_$ViewNum.ply`. The camera poses for these views and the resized binary masks and RGB images are stored in `cam_$ShapeName_$ViewNum.txt`, `$ShapeName_Mask_$ViewNum_binary`, and `$ShapeName_RGB_$ViewNum` for evaluation of the proposed method.

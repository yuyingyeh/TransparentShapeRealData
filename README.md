# TransparentShapeRealData
The real transparent shape reconstructed and ground truth scanned meshes, evaluation code, and data creation pipeline for [Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/), CVPR'20
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/github/TransShape.gif)

## Download Real Data
### Pre-processed Real Data for Evaluation on Our Network
We have pre-processed real transparent shape and image data used in our paper. [[Download]](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/dataset/RealData.zip). (112.3MB)
* Shapes/real/
  * The geometry, camera position and scene configuration files used to create the dataset. 
* ImagesReal/real/
  * The rendered images, two-bounce normal and the final reconstructed meshes of the 5-12 view reconstruction. 
* Envmaps/real/
  * The environment maps we created using the [createRealData pipeline](https://github.com/yuyingyeh/TransparentShapeRealData/tree/master/createRealData).
  
### Reconstructed and Ground Truth Scanned Meshes for Direct Evaluation
We provide the reconstructed and the ground truth scanned meshes for direct evaluation. Please use these aligned reconstruced and ground truth mesh pairs to perform baseline comparisons. [[Download]](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/dataset/RealMesh.zip) (191MB)

## Process Your Own Captured Data
Please see [createRealData](https://github.com/yuyingyeh/TransparentShapeRealData/tree/master/createRealData) folder.

## Evaluation Code on Real Shapes
Please see [evalRealData](https://github.com/yuyingyeh/TransparentShapeRealData/tree/master/evalRealData) folder.

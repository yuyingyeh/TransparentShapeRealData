import numpy as np
import os
import cv2
from skimage import measure
import trimesh as trm
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
# code for reconstruct visual hull mesh

parser = argparse.ArgumentParser()
parser.add_argument('--scene', default='Scene')
parser.add_argument('--shapeName', default='Mouse')
parser.add_argument('--nViews', type=int, default=10)
parser.add_argument('--asDataset', action='store_true', help='use this tag to change output path to real data folder')
parser.add_argument('--shapeId', type=int, default=0, help='activate when asDataset is true, indicate shape id for the shape')         
opt = parser.parse_args()

expDir = opt.scene
imgFile = os.path.join(expDir, 'sparse/0/images.txt')
camFile = os.path.join(expDir, 'sparse/0/cameras.txt')
animalName = opt.shapeName
nViews = opt.nViews
imgFolder = os.path.join(expDir, 'All_Imgs') # RGB folder to copy selected RGB images
maskFolder = os.path.join(expDir, '{}_Mask_{}'.format(animalName, nViews) )
maskFiles = sorted(glob.glob(os.path.join(maskFolder, '*.png')) )
numMask = len(maskFiles)
assert numMask>0, 'Mask folder {} is empty!'.format(maskFolder)
assert nViews == numMask, 'Number of views should equal to number of masks!'

meshName = os.path.join(expDir, '{}_visualHull_{}.ply'.format(animalName, nViews) )
meshSubdName = os.path.join(expDir, '{}_visualHullSubd_{}.ply'.format(animalName, nViews) )
if not opt.asDataset:
    outputCamFile = os.path.join(expDir, 'cam_{}_{}.txt'.format(animalName, nViews) )
    outputMaskFolder = os.path.join(expDir, '{}_Mask_{}_binary'.format(animalName, nViews) )
    outputRGBFolder = os.path.join(expDir, '{}_RGB_{}'.format(animalName, nViews) )
else:
    outputCamFile = os.path.join('Shapes', 'real', 'Shape__%d' % opt.shapeId, 'cam%d.txt' % nViews)
    outputMaskFolder = os.path.join('ImagesReal', 'real', 'Shape__%d' % opt.shapeId)
    outputRGBFolder = outputMaskFolder
sourceRGBFolder = os.path.join(expDir, animalName)
resizeRGBimH = 360
resizeRGBimW = 480

buildVisualHull = True
plot3Dcam = False
checkMode = False
checkFolder = maskFolder.replace(expDir, expDir+'check_')
if checkMode == True and os.path.exists(checkFolder) == False:
    os.makedirs(checkFolder)

outputMode = True
writeOutputCamFile = outputMode
saveBinaryMask = outputMode
saveRGB = outputMode

if writeOutputCamFile == True and os.path.exists(os.path.dirname(outputCamFile)) == False:
    os.makedirs(os.path.dirname(outputCamFile))
if saveBinaryMask == True and os.path.exists(outputMaskFolder) == False:
    os.makedirs(outputMaskFolder)
if saveRGB == True and os.path.exists(outputRGBFolder) == False:
    os.makedirs(outputRGBFolder)

# Read camera file
c = 0
with open(camFile, 'r') as camParams:
    for p in camParams.readlines():
        c += 1
        if c <= 3: # skip comments
            continue
        else:
            line = p.strip().split(' ')
            imgW, imgH = int(line[2]), int(line[3])
            f = float(line[4])
            cxp, cyp = int(line[5]), int(line[6])
            break

def quaternionToRotation(Q):
    # q = a + bi + cj + dk
    a = float(Q[0])
    b = float(Q[1])
    c = float(Q[2])
    d = float(Q[3])

    R = np.array([[2*a**2-1+2*b**2, 2*b*c+2*a*d,     2*b*d-2*a*c],
                  [2*b*c-2*a*d,     2*a**2-1+2*c**2, 2*c*d+2*a*b],
                  [2*b*d+2*a*c,     2*c*d-2*a*b,     2*a**2-1+2*d**2]])
    return np.transpose(R)

# Read images file
c = 0
camDict = {}
with open(imgFile, 'r') as camPoses:
    for cam in camPoses.readlines():
        c += 1
        if c <= 3: # skip comments
            continue
        elif c == 4:
            numImg = int(cam.strip().split(',')[0].split(':')[1])
            print('Number of images:', numImg)
        else:
            if c % 2 == 1:
                line = cam.strip().split(' ')
                R = quaternionToRotation(line[1:5])
                paramDict = {}
                paramDict['Rot'] = R
                paramDict['Trans'] = np.array([float(line[5]), float(line[6]), float(line[7])])
                paramDict['Origin'] = -np.matmul(np.transpose(R), paramDict['Trans'])
                paramDict['Target'] = R[2, :] + paramDict['Origin']
                paramDict['Up'] = -R[1, :]
                paramDict['cId'] = int(line[0])
                name = line[9]
                paramDict['imgName'] = name
                nameId = name.split('_')[2].split('.')[0]
                camDict[nameId] = paramDict

# initialize visual hull voxels
resolution = 256

minX, maxX = -1.7, 1.7
minY, maxY = -1.7, 1.7
minZ, maxZ = -1.7, 1.7

y, x, z = np.meshgrid(
        np.linspace(minX, maxX, resolution),
        np.linspace(minY, maxY, resolution),
        np.linspace(minZ, maxZ, resolution)
        )

x = x[:, :, :, np.newaxis]
y = y[:, :, :, np.newaxis]
z = z[:, :, :, np.newaxis]
coord = np.concatenate([x, y, z], axis=3)
volume = -np.ones(x.shape).squeeze()

############################
# Start to build the voxel #
############################
if plot3Dcam == True:
    centers = []
    lookups = []
    ups = []
    ids = []

if writeOutputCamFile == True:
    if os.path.exists(outputCamFile) == True:
        print("{} exists!!! Insert y if want to overwrite:".format(outputCamFile))
        overwrite = input()
        if overwrite == 'y':
            os.remove(outputCamFile)
        else:
            assert False, "{} exists!!!".format(outputCamFile)

    with open(outputCamFile, 'w') as cam:
        cam.write('{}\n'.format(numMask))

for i in range(numMask):
    print('Processing {}/{} mask:'.format(i+1, numMask))
    #print(os.path.join(maskFolder, maskFiles[i]))
    #seg = cv2.imread(os.path.join(maskFolder, maskFiles[i]), cv2.IMREAD_UNCHANGED)[:,:,3]
    seg = cv2.imread(maskFiles[i], cv2.IMREAD_UNCHANGED)[:,:,3]
    baseName = os.path.basename(maskFiles[i])
    if saveBinaryMask == True:
        print(os.path.join(outputMaskFolder, baseName))
        segResize = cv2.resize(seg, (resizeRGBimW, resizeRGBimH), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(outputMaskFolder, 'seg_{}.png'.format(i+1)), segResize)
    mask = seg.reshape(imgH * imgW)

    imgId = baseName.split('_')[2].split('.')[0]

    if saveRGB == True:
        rgbName = os.path.join(sourceRGBFolder, baseName).replace('png', 'jpg')
        assert os.path.exists(rgbName), 'RGB image {} does not exist!'.format(rgbName)
        rgbImg = cv2.imread(rgbName)
        rgbImg = cv2.resize(rgbImg, (resizeRGBimW, resizeRGBimH), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(outputRGBFolder, 'im_{}.png'.format(i+1)), rgbImg)

    '''
    # If given extrinsic
    Rot = camDict[imgId]['Rot']
    Trans = camDict[imgId]['Trans']
    Trans = Trans.reshape([1, 1, 1, 3, 1])
    coordCam = np.matmul(Rot, np.expand_dims(coord, axis=4)) + Trans
    '''
    # If given Origin, Target, Up
    origin = camDict[imgId]['Origin']
    C = origin

    target = camDict[imgId]['Target']
    up = camDict[imgId]['Up']

    #print('Origin:',origin, ' Target:', target, ' Up:', up)

    if plot3Dcam == True:
        centers.append(origin)
        lookups.append((target-origin))
        ups.append(up)
        ids.append(imgId)

    if writeOutputCamFile == True:
        with open(outputCamFile, 'a') as cam:
            cam.write('{} {} {}\n'.format(origin[0], origin[1], origin[2]))
            cam.write('{} {} {}\n'.format(target[0], target[1], target[2]))
            cam.write('{} {} {}\n'.format(up[0], up[1], up[2]))

    if buildVisualHull == True:

        yAxis = up / np.sqrt(np.sum(up * up))
        zAxis = target - origin
        zAxis = zAxis / np.sqrt(np.sum(zAxis * zAxis))
        xAxis = np.cross(zAxis, yAxis)
        xAxis = xAxis / np.sqrt(np.sum(xAxis * xAxis))
        Rot = np.stack([xAxis, yAxis, zAxis], axis=0)
        coordCam = np.matmul(Rot, np.expand_dims(coord - C, axis=4))


        coordCam = coordCam.squeeze(4)
        xCam = coordCam[:, :, :, 0] / coordCam[:, :, :, 2]
        yCam = coordCam[:, :, :, 1] / coordCam[:, :, :, 2]

        xId = xCam * f + cxp
        yId = -yCam * f + cyp

        xInd = np.logical_and(xId >= 0, xId < imgW-0.5)
        yInd = np.logical_and(yId >= 0, yId < imgH-0.5)
        imInd = np.logical_and(xInd, yInd)

        xImId = np.round(xId[imInd]).astype(np.int32)
        yImId = np.round(yId[imInd]).astype(np.int32)

        maskInd = mask[yImId * imgW + xImId]

        volumeInd = imInd.copy()
        volumeInd[imInd == 1] = maskInd

        volume[volumeInd == 0] = 1

        print('Occupied voxel: %d' % np.sum((volume > 0).astype(np.float32)))

        verts, faces, normals, _ = measure.marching_cubes_lewiner(volume, 0)
        print('Vertices Num: %d' % verts.shape[0])
        print('Normals Num: %d' % normals.shape[0])
        print('Faces Num: %d' % faces.shape[0])

        axisLen = float(resolution-1) / 2.0
        verts = (verts - axisLen) / axisLen * 1.7
        mesh = trm.Trimesh(vertices = verts, vertex_normals = normals, faces = faces)

        if checkMode == True:
            print('Export mesh ', imgId, ' cId:', camDict[imgId]['cId'])
            mesh.export(os.path.join(checkFolder, os.path.basename(meshName.replace('.ply', '_{}.ply'.format(imgId)))))

print('Export final mesh !')
mesh.export(meshName)

if plot3Dcam == True:
    #code for visualize lookup and up vectors
    centers = np.stack(centers, axis=0)
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]
    lookups = np.stack(lookups, axis=0)
    lookU = lookups[:, 0]
    lookV = lookups[:, 1]
    lookW = lookups[:, 2]
    ups = np.stack(ups, axis=0)
    upsU = ups[:, 0]
    upsV = ups[:, 1]
    upsW = ups[:, 2]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.quiver(x, y, z, lookU, lookV, lookW, cmap='Reds')
    #ax.quiver(x, y, z, upsU, upsV, upsW, cmap='Blues')
    ax.scatter(x, y, z, c='r', marker='o')
    for i in range(len(ids)):
        ax.text(x[i], y[i], z[i], ids[i])

    plt.show()

cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i %s -o %s -om vn -s remeshVisualHullNew.mlx' % \
        ((meshName), (meshSubdName))

os.system(cmd)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
from scipy import ndimage
from scipy import stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scene', default='Scene')
parser.add_argument('--showStepByStep', action='store_true', help='use this flag to output intermediate results for debugging')
parser.add_argument('--param', type=int, default=0, help='select pre-tuned parameters')
opt = parser.parse_args()

expDir = opt.scene
imgFile = os.path.join(expDir, 'sparse/0/images.txt')
camFile = os.path.join(expDir, 'sparse/0/cameras.txt')
#ptsFile = os.path.join(expDir, 'sparse/0/points3D.txt')
imgFolder = os.path.join(expDir, 'MirrorBall_imgs')
ballLocFile = os.path.join(expDir, 'ballLoc.txt')
envSavePath = os.path.join(expDir, 'env.png')

showStepByStep = opt.showStepByStep
checkFolder = os.path.join(expDir, 'checkEnv')
if showStepByStep == True and os.path.exists(checkFolder) == False:
    os.makedirs(checkFolder)

envH = 256
envW = 512

envParamDict = {
        'sfm_4f/':   [0.95, 0.3, 0.1 , 0.8, 0.2, 0.3],
        'sfm_3127/': [0.95, 0.2, 0.35, 0.7, 0.2, 0.3]
        }
# Two choices of parameters, require tuning for better results
if opt.param == 0:
    param = envParamDict['sfm_4f/']
elif opt.param == 1:
    param = envParamDict['sfm_3127/']
rTHout = param[0]
rTHin = param[1]
topTH = param[2]
bottomTH = param[3]
leftTH = param[4]
rightTH = param[5]

sLow = 5
sUp = 15
nScaleStep = 1001

nFilter = 10
filterSize = 10

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
#assert c>4, "More than one camera!"

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

# Read 3d points file
# c = 0
# px = []
# py = []
# pz = []
# with open(ptsFile, 'r') as pts3D:
#     for p in pts3D.readlines():
#         c += 1
#         if c <= 3: # skip comments
#             continue
#         else:
#             line = p.strip().split(' ')
#             ppx = float(line[1])
#             ppy = float(line[2])
#             ppz = float(line[3])
#             if np.sqrt(ppx**2 + ppy**2 + ppz**2) < 4.9:
#                 px.append(float(line[1]))
#                 py.append(float(line[2]))
#                 pz.append(float(line[3]))

# check results
#assert len(quaternionList)==numImg, "Number of images not consistent!"
#assert len(np.unique(cameraIdList))==1, "Multiple cameras!"

'''
xc = []
yc = []
zc = []
maxC = 0
maxCdist = 0
# Get camera centers
for i in range(numImg):
    Q = quaternionList[i]
    R = quaternionToRotation(Q)
    t = np.array([float(translationList[i][j]) for j in range(3)])
    C = -np.matmul(np.transpose(R), t)
    Cnorm = np.sqrt(np.sum(C**2))
    if Cnorm > maxC:
        maxC = Cnorm
    if i == 0:
        c1 = C
    else:
        cdist = np.sqrt(np.sum((C-c1)**2))
        if maxCdist < cdist:
            maxCdist = cdist
    xc.append(C[0])
    yc.append(C[1])
    zc.append(C[2])
'''

ballLocDict = {}
with open(ballLocFile, 'r') as locFile:
    for line in locFile.readlines():
        pos = line.strip().split(' ')
        # imgId top bottom left right
        ballLocDict[pos[0]] = [int(pos[1]), int(pos[2]), int(pos[3]), int(pos[4])]
#numBallLoc = len(ballLocDict)

# find mirror ball center
minSTD = np.inf
scaleArray = np.linspace(sLow, sUp, num=nScaleStep)
stdList = []
for s in scaleArray:
    ballx = []
    bally = []
    ballz = []
    rBallW = 1 / 1000 * 76.2/2 * s # (m) * scale
    centerList = []
    for imgId, pos in ballLocDict.items():
        R = camDict[imgId]['Rot']
        t = camDict[imgId]['Trans']
        top, bottom, left, right = pos[0], pos[1], pos[2], pos[3]
        cxBall, cyBall = (right+left)/f/2, (top+bottom)/f/2
        cx, cy = cxp / f, cyp / f
        rxBall, ryBall = (right-left)/f/2, (bottom-top)/f/2,
        dBallCenterImg = np.sqrt((cxBall-cx)**2 + (cyBall-cy)**2 + 1**2)
        # similar triangle
        dxBallCenterW = rBallW / rxBall * np.sqrt(rxBall**2+dBallCenterImg**2)
        dyBallCenterW = rBallW / ryBall * np.sqrt(ryBall**2+dBallCenterImg**2)
        dBallCenterW = (dxBallCenterW + dyBallCenterW) / 2
        ballCenter = np.array([(cxBall-cx), (cyBall-cy), 1]) / dBallCenterImg * dBallCenterW # in unit
        translatedCenter = np.matmul(np.transpose(R), ballCenter-t)
        centerList.append(translatedCenter)
        ballx.append(translatedCenter[0])
        bally.append(translatedCenter[1])
        ballz.append(translatedCenter[2])
        # test if center is correct
        '''
        ballCenterC = np.matmul(R, translatedCenter) + t
        ballCenteru, ballCenterv = ballCenterC[0]/ballCenterC[2], ballCenterC[1]/ballCenterC[2]
        print(ballCenteru, ballCenterv)
        im = plt.imread(imgFolder+'/'+nameList[i])
        implot = plt.imshow(im)
        plt.scatter(x=[ballCenteru*f+cxp], y=[ballCenterv*f+cyp])
        plt.show()
        '''
    centers = np.stack(centerList, axis=0)
    std = np.std(centers, axis=0)
    std = np.sqrt(np.sum(std**2))
    stdList.append(std)
    if std < minSTD:
        minSTD = std
        scale = s
        ballCenterW = np.mean(centers, axis=0)

print('Mirror ball center: ', ballCenterW)
print('Optimal scale:', scale)

#fig, ax = plt.subplots()
#ax.plot(scaleArray, stdList)
#plt.show()

'''
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(xc, yc, zc, c='r', marker='o')
#ax.scatter(xw, yw, zw, c='g', marker='.')
ax.scatter(ballx, bally, ballz, c='b', marker='x')
ax.scatter(px, py, pz, c='k', marker='.')
ax.scatter(ballCenterW[0], ballCenterW[1], ballCenterW[2], c='m', marker='x')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
'''
print('Creating env map for {}'.format(expDir))
# After finding camera center, reconstruct envionment map
# First step: find valid range of rays shooting from the camera
rBallW = 1 / 1000 * 76.2/2 * scale # (m) * scale
envMapR = np.zeros((envH*envW))
envMapG = np.zeros((envH*envW))
envMapB = np.zeros((envH*envW))
envMapDist = np.ones((envH*envW))*np.inf
#for imgId, _ in ballLocDict.items():
ballImgFiles = sorted(glob.glob(imgFolder+'/*.jpg'))
numImg = len(ballImgFiles)
for i in range(numImg):
    imgId = os.path.basename(ballImgFiles[i]).split('_')[2].split('.')[0]
    R = camDict[imgId]['Rot']
    t = camDict[imgId]['Trans']
    C = camDict[imgId]['Origin']
    target = camDict[imgId]['Target']
    up = camDict[imgId]['Up']
    ballCenterC = np.matmul(R, ballCenterW) + t
    ballCenteru, ballCenterv = ballCenterC[0]/ballCenterC[2]*f+cxp, ballCenterC[1]/ballCenterC[2]*f+cyp
    # create a mask for valid pixels
    theta = np.arcsin(rBallW/np.sqrt(np.sum(ballCenterC**2)))
    rBallImg = f * np.tan(theta)
    u = np.arange(imgW)
    v = np.arange(imgH)
    uu, vv = np.meshgrid(u, v)
    m1 = ((uu-ballCenteru)**2 + (vv-ballCenterv)**2) < (rBallImg * rTHout)**2
    m2 = ((uu-ballCenteru)**2 + (vv-ballCenterv)**2) > (rBallImg * rTHin)**2
    mask1 = m1 * m2
    # topTH, bottomTH, leftTH, rightTH
    m3 = (uu > (ballCenteru - rBallImg * leftTH)) * (uu < (ballCenteru + rBallImg * rightTH))
    m4 = (vv > (ballCenterv - rBallImg * topTH)) * (vv < (ballCenterv + rBallImg * bottomTH))
    mask2 = np.logical_not(m3 * m4)
    maskOri = np.logical_and(mask1, mask2)
    mask = np.reshape(maskOri, (imgW*imgH))
    dist = np.sqrt(((uu-ballCenteru)**2 + (vv-ballCenterv)**2))
    dist = np.reshape(dist, (imgW*imgH))[mask] # dist to ball center on each pixel
    # Compute view direction in world coord.
    xx, yy = np.meshgrid(np.linspace(-1, 1, imgW), np.linspace(-1, 1, imgH))
    xx = np.reshape(xx * imgW/2/f, (imgW*imgH))[mask] # 1-d
    yy = np.reshape(yy * imgH/2/f, (imgW*imgH))[mask] # 1-d
    zz = np.ones((xx.size), dtype=np.float32) # 1-d
    v = np.stack([xx, yy, zz], axis=1).astype(np.float32) # m x 3
    v = v / np.maximum(np.sqrt(np.sum(v**2, axis=1))[:, np.newaxis], 1e-6)
    v = np.expand_dims(v, axis=2) # m x 3 x 1
    # Transform to world coord from cam params
    vW = np.matmul(np.expand_dims(np.transpose(R), axis=0), v).squeeze(2) # m x 3
    '''
    zAxis = C - target
    yAxis = up
    xAxis = np.cross(yAxis, zAxis)
    xAxis = xAxis / np.sqrt(np.sum(xAxis**2))
    rotMat = np.expand_dims(np.stack([xAxis, yAxis, zAxis], axis=1), axis=0) # m x 3 x 3
    vW = np.matmul(rotMat, v).squeeze(2) # m x 3
    '''
    # Compute normal by 3d geometry
    pa = np.sum(vW**2, axis=1) # m
    pb = np.sum(2 * vW * (C - ballCenterW), axis=1) # m
    pc = np.sum((C - ballCenterW)**2) - rBallW**2 # 1
    #t1 = (-pb + np.sqrt(pb**2-4*pa*pc))/(2*pa)
    root = (-pb - np.sqrt(pb**2-4*pa*pc))/(2*pa) # m
    normal = C + vW * np.expand_dims(root, axis=1) - ballCenterW # m x 3
    normal = normal / np.expand_dims(np.sqrt(np.sum(normal**2, axis=1)), axis=1) # m x 3
    # Reflection
    cos_theta = np.sum(vW * (-normal), axis=1) # m
    r_p = vW + normal * np.expand_dims(cos_theta, axis=1) # m x 3
    r_p_norm = np.sum(r_p**2, axis=1)
    r_i = np.expand_dims(np.sqrt(1 - r_p_norm), axis=1) * normal # m x 3
    vWr = r_p + r_i
    vWr = vWr / np.expand_dims(np.sqrt(np.sum(vWr**2, axis=1)), axis=1)
    # Compute angle from reflected directions to env (u,v)
    theta = np.arccos(vWr[:, 1]) # m
    phi = np.arctan2(vWr[:, 0], vWr[:, 2]) # m
    envv = np.rint(theta / np.pi * (envH)).astype(np.int64) % envH # m
    envu = np.rint((-phi / np.pi / 2.0 + 0.5) * (envW)).astype(np.int32) % envW # m
    envidx = envv * envW + envu # m, discrete idx
    # Remove duplicate env idx and select idx near ball center
    color = np.reshape(plt.imread(imgFolder+'/'+camDict[imgId]['imgName']), (imgW*imgH, 3))[mask, :] # m x 3
    envidx, uniqueIdx  = np.unique(envidx, return_index=True) # m2
    dist = dist[uniqueIdx] # m2
    color = color[uniqueIdx, :].astype(envMapR.dtype) # m2 x 3

    updateMask = dist < envMapDist[envidx] # m2-d bool, assume n True
    colorUpdate = color[updateMask, :] # n x 3
    envidxUpdate = envidx[updateMask] # n
    distUpdate = dist[updateMask] # n
    # Put color onto envMap, first check if idx is closer to ball center
    envMapR[envidxUpdate] = colorUpdate[:, 0]
    envMapG[envidxUpdate] = colorUpdate[:, 1]
    envMapB[envidxUpdate] = colorUpdate[:, 2]
    envMapDist[envidxUpdate] = distUpdate

    if showStepByStep == True:
        # check correct range on ball
        checkIm = plt.imread(imgFolder+'/'+camDict[imgId]['imgName'])
        maskedIm = checkIm*np.logical_not(np.expand_dims(maskOri, axis=2))
        #checkimplot = plt.imshow(checkIm*np.logical_not(np.expand_dims(maskOri, axis=2)))
        #plt.show()
        plt.imsave(os.path.join(checkFolder, 'ballMask_{}.png'.format(imgId)), maskedIm)

        # show current env map
        envMapR0 = np.zeros((envH*envW))
        envMapG0 = np.zeros((envH*envW))
        envMapB0 = np.zeros((envH*envW))
        envMapR0[envidx] = color[:, 0]
        envMapG0[envidx] = color[:, 1]
        envMapB0[envidx] = color[:, 2]
        envMap0 = np.reshape(np.stack([envMapR0, envMapG0, envMapB0], axis=1), (envH, envW, 3))
        #implot = plt.imshow(envMap0.astype(np.long))
        #plt.show()
        plt.imsave(os.path.join(checkFolder, 'env_{}.png'.format(imgId)), envMap0.astype(np.uint8))

envMap = np.reshape(np.stack([envMapR, envMapG, envMapB], axis=1), (envH, envW, 3))
plt.imsave(envSavePath, envMap.astype(np.uint8))

print('Environment map has been created!!!')


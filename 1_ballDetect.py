import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scene', default='Scene')
parser.add_argument('--isPrecise', action='store_true', help='use this flag to manually select precise circle prediction')
opt = parser.parse_args()

#Loading Image
expDir = opt.scene
imgFolder = os.path.join(expDir, 'MirrorBall_Imgs')
imgFile = os.path.join(expDir, 'sparse/0/images.txt')
circleFile = os.path.join(expDir, 'ballLoc.txt')

# Read images file
c = 0
camDict = {}
quaternionList = []
translationList = []
cameraIdList = []
nameList = []
nameIdList = []
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
                paramDict = {}
                name = line[9]
                paramDict['imgName'] = name
                nameId = name.split('_')[2].split('.')[0]
                camDict[nameId] = paramDict

if os.path.exists(circleFile):
    assert False, 'circle location file already exists!'

ballImgFiles = sorted(glob.glob(imgFolder+'/*.jpg'))
numImg = len(ballImgFiles)
for i in range(numImg):
    img = cv2.imread(ballImgFiles[i])
    img_org = img.copy()
    img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    plt.rcParams["figure.figsize"] = (16,9)
    plt.imshow(img,cmap='gray')

    #Applying Blur to the image

    img = cv2.GaussianBlur(img, (21,21), cv2.BORDER_DEFAULT)
    plt.rcParams["figure.figsize"] = (16,9)
    plt.imshow(img,cmap='gray')

    H, W = img.shape
    all_circs = cv2.HoughCircles(img , cv2.HOUGH_GRADIENT,1, 500,param1 = 20 , param2 = 30, minRadius=int(H/30), maxRadius=int(H/6))
    cy, cx = H/2.0, W/2.0
    # check if its radius is smaller than 1/6 of imgH and center is close to center
    all_circs = np.squeeze(all_circs, axis=0)
    select1 = all_circs[:, 2] < (H/6)
    select2 = (all_circs[:, 0] < (cx+W*0.05)) * (all_circs[:, 0] > (cx-W*0.05))
    select3 = (all_circs[:, 1] < (cy+H*0.1)) * (all_circs[:, 1] > (cy-H*0.1))
    select = select1 * select2 * select3
    select_circs = all_circs[select]
    imgId = os.path.basename(ballImgFiles[i]).split('_')[2].split('.')[0]
    print(camDict[imgId]['imgName'], end='')
    if select_circs.shape[0]<1:
        print(' Ball detection failed!')
    elif select_circs.shape[0]>1:
        print(' Find two circles near the center!')
    else:
        print(' Find one circle in the center!')
        print('Img ID: ', imgId)
        coor = select_circs[0] # x, y, radius
        top = int(coor[1]-coor[2])
        bottom = int(coor[1]+coor[2])
        left = int(coor[0]-coor[2])
        right = int(coor[0]+coor[2])
        print('Top:', top, ' Bottom: ', bottom, end=' ')
        print('Left:', left, ' Right: ', right)

        if opt.isPrecise:
            circle = np.uint16(np.around(coor))
            cv2.circle(img_org, (coor[0], coor[1]), coor[2], (200, 0, 0), 1)
            plt.imshow(img_org)
            plt.show()

            print('Insert y if accept this circle estimation, otherwise input anything:')
            decision = input()
        else:
            decision = 'y'

        if decision == 'y':
            with open(circleFile, 'a') as locFile:
                locFile.write('{} {} {} {} {}\n'.format(imgId, top, bottom, left, right))

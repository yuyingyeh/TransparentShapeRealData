import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
import models
import torchvision.utils as vutils
import utils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.io import loadmat
import math
import cv2
import torch.nn.functional as F
import os.path as osp
import h5py
import time

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='/home/zhl/CVPR20/TransparentShape/Data/ImagesReal', help='path to images' )
parser.add_argument('--shapeRoot', default='/home/zhl/CVPR20/TransparentShape/Data/Shapes/', help='path to images' )
parser.add_argument('--mode', default='real', help='whether to train the network or test the network')
parser.add_argument('--shapeId', default=None, help='if not none, eval on selected shapeId')
parser.add_argument('--experiment', default=None, help='the path to store samples and models' )
parser.add_argument('--optimRoot', default=None, help='the path to store the optimization samples')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=10, help='the number of epochs for training' )
parser.add_argument('--iterationNum', type=int, default=500, help='the number of iterations for optimization' )
parser.add_argument('--batchSize', type=int, default=17, help='input batch size' )
parser.add_argument('--imageHeight', type=int, default=192, help='the height / width of the input image to network' )
parser.add_argument('--imageWidth', type=int, default=256, help='the height / width of the input image to network' )
parser.add_argument('--envHeight', type=int, default=256, help='the height / width of the input envmap to network' )
parser.add_argument('--envWidth', type=int, default=512, help='the height / width of the input envmap to network' )
parser.add_argument('--volumeSize', type=int, default=64, help='the size of the volume' )
# The parameters
parser.add_argument('--camNum', type=int, default=10, help='the number of views to create the visual hull' )
parser.add_argument('--sampleNum', type=int, default=1, help='the sample num for the cost volume')
parser.add_argument('--shapeStart', type=int, default=0, help='the start id of the shape')
parser.add_argument('--shapeEnd', type=int, default=3000, help='the end id of the shape')
parser.add_argument('--isAddCostVolume', action='store_true', help='whether to use cost volume or not' )
parser.add_argument('--poolingMode', type=int, default=2, help='0: maxpooling, 1: average pooling 2: learnable pooling' )
parser.add_argument('--isAddVisualHull', action='store_true', help='whether to build visual hull into training' )
parser.add_argument('--isMultiScale', action='store_true', help='whether to use mutli scale or not')
parser.add_argument('--scaleNum', default=2, type=int, help='how many times for downsampling')
parser.add_argument('--isH5py', action='store_true', help='whether to write the format in h5py')
# The rendering parameters
parser.add_argument('--eta1', type=float, default=1.0003, help='the index of refraction of air' )
parser.add_argument('--eta2', type=float, default=1.4723, help='the index of refraction of glass' )
parser.add_argument('--fov', type=float, default=63.4, help='the field of view of camera' )
# The loss parameters
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for normal' )
parser.add_argument('--renderWeight', type=float, default=0.0, help='the weight for rendering loss')
parser.add_argument('--intermediateWeight', type=float, default=0.01, help='the weight of intermediate supervision')
# The gpu setting
parser.add_argument('--cuda', action='store_true', help='enables cuda' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )

opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]
#opt.dataRoot = opt.dataRoot + '%d' % opt.camNum

nw = opt.normalWeight
rw = opt.renderWeight
iw = opt.intermediateWeight
opt.dataRoot = osp.join(opt.dataRoot, opt.mode )
opt.shapeRoot = osp.join(opt.shapeRoot, opt.mode )

if opt.batchSize is None:
    opt.batchSize = opt.camNum

####################################
# Initialize the angles
if opt.camNum == 5:
    thetas = [0, 25, 25, 25]
    phis = [0, 0, 120, 240]
elif opt.camNum == 10:
    thetas = [0, 15, 15, 15]
    phis = [0, 0, 120, 240]
elif opt.camNum == 20:
    thetas = [0, 10, 10, 10]
    phis = [0, 0, 120, 240]

thetaJitters = [0, 0, 0, 0]
phiJitters = [0, 0, 0, 0]

thetas = np.array(thetas ).astype(np.float32 ) / 180.0 * np.pi
phis = np.array(phis ).astype(np.float32 ) / 180.0 * np.pi
thetaJitters = np.array(thetaJitters ).astype(np.float32 ) / 180.0 * np.pi
phiJitters = np.array(phiJitters ).astype(np.float32 ) / 180.0 * np.pi
angleNum = thetas.shape[0]

if opt.experiment is None:
    opt.experiment = "check%d_normal_nw%.2f_rw%.2f" % (opt.camNum, nw, rw)
    if opt.isAddCostVolume:
        if opt.poolingMode == 0:
            opt.experiment +=  '_volume_sp%d_an%d_maxpool' % (opt.sampleNum, angleNum )
        elif opt.poolingMode == 1:
            opt.experiment += '_volume_sp%d_an%d_avgpool' % (opt.sampleNum, angleNum )
        elif opt.poolingMode == 2:
            opt.experiment += '_volume_sp%d_an%d_weigtedSum' % (opt.sampleNum, angleNum )
        else:
            print("Wrong: unrecognizable pooling mode." )
            assert(False )
    if opt.isAddVisualHull:
        opt.experiment += '_vh_iw%.2f' % iw


if not osp.isdir(opt.experiment ):
    print('Wrong: the model %s does not exist!' % opt.experiment )
    assert(False )

opt.optimRoot = opt.experiment.replace('check', 'optimization_%s' % (opt.mode) )
opt.optimRoot = 'temp_countTime'
if opt.isMultiScale:
    opt.optimRoot += '_mc%d' % opt.scaleNum
os.system('mkdir {0}'.format( opt.optimRoot ) )
os.system('cp *.py %s' % opt.optimRoot )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda" )
colorMapFile = './colormap.mat'
colormap = loadmat(colorMapFile)['cmap']
colormap = torch.from_numpy(colormap).cuda()

####################################
# Initialize Network
encoder = models.encoder(isAddCostVolume = opt.isAddCostVolume )
for param in encoder.parameters():
    param.requires_grad = False
encoder.load_state_dict(torch.load('{0}/encoder_{1}.pth'.format(opt.experiment, opt.nepoch-1 ) ) )

decoder = models.decoder(isAddVisualHull = opt.isAddVisualHull )
for param in decoder.parameters():
    param.requires_grad = False
decoder.load_state_dict(torch.load('{0}/decoder_{1}.pth'.format(opt.experiment, opt.nepoch-1 ) ) )

normalFeature = models.normalFeature()
for param in normalFeature.parameters():
    param.requires_grad = False
normalFeature.load_state_dict(torch.load('{0}/normalFeature_{1}.pth'.format(opt.experiment, opt.nepoch-1) ) )

normalPool = Variable(torch.ones([1, angleNum * angleNum, 1, 1, 1], dtype=torch.float32 ) )
normalPool.requires_grad = False
if opt.isAddCostVolume and opt.poolingMode == 2:
    normalPool.data.copy_(torch.load('{0}/normalPool_{1}.pth'.format(opt.experiment, opt.nepoch-1) ) )

# Other modules
renderer = models.renderer(eta1 = opt.eta1, eta2 = opt.eta2,
        isCuda = opt.cuda, gpuId = opt.gpuId,
        batchSize = opt.batchSize,
        fov = opt.fov,
        imWidth=opt.imageWidth, imHeight = opt.imageHeight,
        envWidth = opt.envWidth, envHeight = opt.envHeight )

if opt.isAddVisualHull:
    encoderVH = models.encoderVH()
    for param in encoderVH.parameters():
        param.requires_grad = False
    encoderVH.load_state_dict(torch.load('{0}/encoderVH_{1}.pth'.format(opt.experiment, opt.nepoch-1 ) ) )

    encoderCamera = models.encoderCamera()
    for param in encoderCamera.parameters():
        param.requires_grad = False
    encoderCamera.load_state_dict(torch.load('{0}/encoderCamera_{1}.pth'.format(opt.experiment, opt.nepoch-1 ) ) )

    decoderVH = models.decoderVH()
    for param in decoderVH.parameters():
        param.requires_grad = False
    decoderVH.load_state_dict(torch.load('{0}/decoderVH_{1}.pth'.format(opt.experiment, opt.nepoch-1 ) ) )

    buildVisualHull = models.buildVisualHull(fov = opt.fov,
            volumeSize = opt.volumeSize )

    rendererVH = models.renderer(eta1 = opt.eta1, eta2 = opt.eta2,
            isCuda = opt.cuda, gpuId = opt.gpuId,
            batchSize = opt.batchSize,
            fov = opt.fov,
            imWidth= int(opt.imageWidth / 2.0), imHeight = int(opt.imageHeight /2.0),
            envWidth = opt.envWidth, envHeight = opt.envHeight )


##############  ######################
# Send things into GPU
if opt.cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    normalFeature = normalFeature.cuda()
    normalPool = normalPool.cuda()
    if opt.isAddVisualHull:
        encoderVH = encoderVH.cuda()
        encoderCamera = encoderCamera.cuda()
        decoderVH = decoderVH.cuda()
####################################

####################################
if opt.isAddCostVolume:
    buildCostVolume = models.buildCostVolume(
            thetas = thetas, phis = phis,
            thetaJitters = thetaJitters,
            phiJitters = phiJitters,
            eta1 = opt.eta1, eta2 = opt.eta2,
            batchSize = opt.batchSize,
            fov = opt.fov,
            imWidth = opt.imageWidth, imHeight = opt.imageHeight,
            envWidth = opt.envWidth, envHeight = opt.envHeight,
            sampleNum = opt.sampleNum )
else:
    buildCostVolume = None

brdfDataset = dataLoader.BatchLoader(
        opt.dataRoot, shapeRoot = opt.shapeRoot,
        imHeight = opt.imageHeight, imWidth = opt.imageWidth,
        envHeight = opt.envHeight, envWidth = opt.envWidth,
        isRandom = True, phase='TEST', rseed = 1,
        isLoadVH = True, isLoadEnvmap = True, isLoadCam = True,
        shapeRs = opt.shapeStart, shapeRe = opt.shapeEnd,
        batchSize = opt.batchSize, isOptim = True )
brdfLoader = DataLoader( brdfDataset, batch_size = 1, num_workers = 0, shuffle = False )

j = 0

normal1ErrsNpList = np.ones( [1, 3], dtype = np.float32 )
meanAngle1ErrsNpList = np.ones([1, 3], dtype = np.float32 )
medianAngle1ErrsNpList = np.ones([1, 3], dtype = np.float32 )
normal2ErrsNpList = np.ones( [1, 3], dtype = np.float32 )
meanAngle2ErrsNpList = np.ones([1, 3], dtype = np.float32 )
medianAngle2ErrsNpList = np.ones([1, 3], dtype = np.float32 )
renderedErrsNpList = np.ones([1, 3], dtype=np.float32 )

if opt.isAddVisualHull and iw > 0.0:
        normal1VHErrsNpList = np.ones( [1, 2], dtype = np.float32 )
        seg1VHErrsNpList = np.ones([1, 2], dtype=np.float32 )
        normal2VHErrsNpList = np.ones( [1, 2], dtype = np.float32 )
        seg2VHErrsNpList = np.ones([1, 2], dtype=np.float32 )

epoch = 0
trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.optimRoot, epoch ), 'w' )

timeSum = 0
countSum = 0
for i, dataBatch in enumerate(brdfLoader):
    j += 1
    '''
    # Load ground-truth from cpu to gpu
    normal1_cpu = dataBatch['normal1'].squeeze(0)
    normal1Batch = Variable(normal1_cpu ).cuda()
    '''
    seg1_cpu = dataBatch['seg1'].squeeze(0 )
    seg1Batch = Variable((seg1_cpu ) ).cuda()
    '''
    normal2_cpu = dataBatch['normal2'].squeeze(0 )
    normal2Batch = Variable(normal2_cpu ).cuda()

    seg2_cpu = dataBatch['seg2'].squeeze(0 )
    seg2Batch = Variable(seg2_cpu ).cuda()
    '''
    # Load the image from cpu to gpu
    im_cpu = dataBatch['im'].squeeze(0 )
    imBatch = Variable(im_cpu ).cuda()

    imBg_cpu = dataBatch['imE'].squeeze(0 )
    imBgBatch = Variable(imBg_cpu ).cuda()

    # Load environment map
    envmap_cpu = dataBatch['env'].squeeze(0 )
    envBatch = Variable(envmap_cpu ).cuda()

    # Load camera parameters
    origin_cpu = dataBatch['origin'].squeeze(0 )
    originBatch = Variable(origin_cpu ).cuda()

    lookat_cpu = dataBatch['lookat'].squeeze(0 )
    lookatBatch = Variable(lookat_cpu ).cuda()

    up_cpu = dataBatch['up'].squeeze(0 )
    upBatch = Variable(up_cpu ).cuda()

    # Load visual hull data
    normal1VH_cpu = dataBatch['normal1VH'].squeeze(0)
    normal1VHBatch = Variable(normal1VH_cpu ).cuda()

    seg1VH_cpu = dataBatch['seg1VH'].squeeze(0 )
    seg1VHBatch = Variable((seg1VH_cpu ) ).cuda()

    normal2VH_cpu = dataBatch['normal2VH'].squeeze(0 )
    normal2VHBatch = Variable(normal2VH_cpu ).cuda()

    seg2VH_cpu = dataBatch['seg2VH'].squeeze(0 )
    seg2VHBatch = Variable(seg2VH_cpu ).cuda()

    # Load the name
    nameBatch = dataBatch['name']

    # Load number of camera view
    viewNum = dataBatch['camNum']

    # Load shape name
    shapeName = dataBatch['shapeName'][0]
    shapeId = int(shapeName.split('_')[-1])
    if opt.shapeId is not None:
        if int(opt.shapeId) != shapeId:
            continue

    batchSize = normal1VHBatch.size(0 )
    ########################################################
    normal1Preds = []
    normal2Preds = []
    renderedImgs = []
    maskPreds = []

    if opt.isAddVisualHull and iw > 0:
        normal1VHPreds = []
        seg1VHPreds = []
        normal2VHPreds = []
        seg2VHPreds = []


    start = time.time()

    refraction, reflection, maskVH = renderer.forward(
            originBatch, lookatBatch, upBatch,
            envBatch,
            normal1VHBatch, normal2VHBatch )
    renderedImgVH = torch.clamp(refraction + reflection, 0, 1)

    # rescale real image
    scale = torch.sum(renderedImgVH*imBatch*(1-maskVH)) / torch.sum(renderedImgVH*renderedImgVH*(1-maskVH))
    renderedImgVH = renderedImgVH * scale

    normal1Preds.append(normal1VHBatch )
    normal2Preds.append(normal2VHBatch )
    renderedImgs.append(renderedImgVH )
    maskPreds.append(1-maskVH)
    errorVH = torch.sum(torch.pow(renderedImgVH - imBatch, 2.0) * seg1Batch, dim=1).unsqueeze(1)

    inputBatch = torch.cat([imBatch, imBgBatch, seg1Batch,
        normal1VHBatch, normal2VHBatch, errorVH, maskVH], dim=1 )
    if opt.isAddCostVolume:
        costVolume = buildCostVolume.forward(imBatch,
                originBatch, lookatBatch, upBatch,
                envBatch, normal1VHBatch, normal2VHBatch,
                seg1Batch )
        volume = normalFeature(costVolume ).unsqueeze(1 )
        volume = volume.view([batchSize, angleNum * angleNum, 64, int(opt.imageHeight/2), int(opt.imageWidth/2)] )
        if opt.poolingMode == 0:
            volume = volume.transpose(1, 4).transpose(2, 3)
            volume = volume.reshape([-1, 64, angleNum, angleNum ] )
            volume = F.max_pool2d(volume, kernel_size = angleNum )
            volume = volume.reshape([batchSize, int(opt.imageWidth/2), int(opt.imageHeight/2), 64] )
            volume = volume.transpose(1, 3)
        elif opt.poolingMode == 1:
            volume = volume.transpose(1, 4 ).transpose(2, 3 )
            volume = volume.reshape([-1, 64, angleNum, angleNum ] )
            volume = F.avg_pool2d(volume, kernel_size = angleNum )
            volume = volume.reshape([batchSize, int(opt.imageWidth/2), int(opt.imageHeight/2), 64] )
            volume = volume.transpose(1, 3 )
        elif opt.poolingMode == 2:
            weight = F.softmax(normalPool, dim=1 )
            volume = torch.sum(volume * normalPool, dim=1 )
        else:
            assert(False )
        x = encoder(inputBatch, volume )
    else:
        x = encoder(inputBatch )


    if opt.isAddVisualHull:
        error = torch.sum(torch.pow(renderedImgs[0] - imBatch, 2), dim=1).unsqueeze(1)
        error = torch.exp(-2 * error ) * seg1Batch
        error = F.adaptive_avg_pool2d(error, [int(0.75*opt.volumeSize ), opt.volumeSize ] )
        mask = F.adaptive_avg_pool2d(seg1Batch, [int(0.75*opt.volumeSize ), opt.volumeSize ] )
        visualHull = buildVisualHull.forward(originBatch, lookatBatch, upBatch, error, mask )
        vhFeature = encoderVH(visualHull )

        yAxis = upBatch / torch.sqrt(torch.sum(upBatch * upBatch, dim=1) ).unsqueeze(1)
        zAxis = lookatBatch - originBatch
        zAxis = zAxis / torch.sqrt(torch.sum(zAxis * zAxis, dim=1) ).unsqueeze(1)
        xAxis = torch.cross(zAxis, yAxis, dim=1 )
        cameraFeature = encoderCamera(torch.cat([xAxis, yAxis, zAxis, originBatch], dim=1) )
        cameraFeature = cameraFeature.reshape(batchSize, 2, 16 * 12, 8, 8, 8)
        cameraFeature1, cameraFeature2 = torch.split(cameraFeature, [1, 1], dim=1 )

        feature1 = torch.sum(torch.sum(torch.sum(cameraFeature1 * vhFeature.unsqueeze(2), dim=3), dim=3), dim=3)
        feature2 = torch.sum(torch.sum(torch.sum(cameraFeature2 * vhFeature.unsqueeze(2), dim=3), dim=3), dim=3)

        feature1 = feature1.reshape([batchSize, int(x.size(1)/2 ), 12, 16] )
        feature1 = F.interpolate(feature1, scale_factor=2, mode='bilinear' )
        feature2 = feature2.reshape([batchSize, int(x.size(1)/2 ), 12, 16] )
        feature2 = F.interpolate(feature2, scale_factor=2, mode='bilinear' )

        combined = torch.cat([x, feature1 / (8*8*8), feature2 / (8*8*8) ], dim=1 )

        if iw > 0:
            normal1VHPred, seg1VHPred = decoderVH(feature1 )
            normal2VHPred, seg2VHPred = decoderVH(feature2 )
            normal1VHPreds.append(normal1VHPred )
            seg1VHPreds.append(seg1VHPred )
            normal2VHPreds.append(normal2VHPred )
            seg2VHPreds.append(seg2VHPred )

        normal1Pred, normal2Pred = decoder( combined )
        refraction, reflection, maskTr = renderer.forward(
                originBatch, lookatBatch, upBatch,
                envBatch,
                normal1Pred, normal2Pred )
        renderedImg = torch.clamp(refraction + reflection, 0, 1)
        normal1Preds.append(normal1Pred )
        normal2Preds.append(normal2Pred )
        renderedImgs.append(renderedImg )

        maskPredOrigin = (1 - maskTr) * seg1Batch

        xPara = Variable(torch.FloatTensor(x.size() ) ).cuda()
        xPara.requires_grad = True
        vhPara = Variable(torch.FloatTensor(vhFeature.size() ) ).cuda()
        vhPara.requires_grad = True
        xPara.data.copy_(x.data )
        vhPara.data.copy_(vhFeature.data )
        featureOptim = optim.Adam([xPara, vhPara ], lr=1e-3, betas=(0.5, 0.999 ) )

        ########################### Optimize Normal 2 ###############################
        minError = 10
        for n in range(0, opt.iterationNum ):
            feature1 = torch.sum(torch.sum(torch.sum(cameraFeature1 * vhPara.unsqueeze(2), dim=3), dim=3), dim=3)
            feature2 = torch.sum(torch.sum(torch.sum(cameraFeature2 * vhPara.unsqueeze(2), dim=3), dim=3), dim=3)

            feature1 = feature1.reshape([batchSize, int(x.size(1)/2 ), 12, 16] )
            feature1 = F.interpolate(feature1, scale_factor=2, mode='bilinear' )
            feature2 = feature2.reshape([batchSize, int(x.size(1)/2 ), 12, 16] )
            feature2 = F.interpolate(feature2, scale_factor=2, mode='bilinear' )

            combined = torch.cat([xPara, feature1 / (8*8*8), feature2 / (8*8*8) ], dim=1 )

            featureOptim.zero_grad()
            normal1Opt, normal2Opt = decoder( combined )
            refraction, reflection, maskTr = renderer.forward(
                    originBatch, lookatBatch, upBatch,
                    envBatch,
                    normal1Opt, normal2Opt )
            renderedImgOpt = torch.clamp(refraction + reflection, 0, 1)

            maskPred = (1- maskTr) * seg1Batch

            if opt.isMultiScale:
                error = []
                pre = renderedImgOpt
                gt = imBatch
                mask = maskPred
                for m in range(0, opt.scaleNum ):
                    pixelNum = torch.sum(mask )
                    error.append(torch.sum(torch.pow(pre - gt, 2) * mask ) / pixelNum )
                    pre = F.adaptive_avg_pool2d(pre, [int(pre.size(2)/2), int(pre.size(3)/2)] )
                    gt = F.adaptive_avg_pool2d(gt, [int(gt.size(2)/2), int(gt.size(3)/2)] )
                    mask = F.adaptive_avg_pool2d(mask, [int(mask.size(2)/2), int(mask.size(3)/2) ] )
                error = sum(error ) / opt.scaleNum
            else:
                pixel2Num = torch.sum(maskPred )
                error = torch.sum( (torch.pow(renderedImgOpt - imBatch, 2) * maskPred ) ) / pixel2Num

            pixel1Num = torch.sum(seg1Batch )
            errorNormal1 = torch.sum(torch.pow(normal1Opt - normal1Pred, 2) * seg1Batch ) / pixel1Num

            errorTotal = error + errorNormal1 * 10
            errorTotal.backward()

            featureOptim.step()

            if n % 20 == 0:
                print('%d/%d: Error %.3f' % (n, opt.iterationNum, errorTotal.data.item() ) )
                if minError > errorTotal.data.item():
                    minError = errorTotal.data.item()
                else:
                    break

        #################### Optimize Normal 1 and 2 ###############################
        for n in range(0, int(opt.iterationNum /5 ) ):
            feature1 = torch.sum(torch.sum(torch.sum(cameraFeature1 * vhPara.unsqueeze(2), dim=3), dim=3), dim=3)
            feature2 = torch.sum(torch.sum(torch.sum(cameraFeature2 * vhPara.unsqueeze(2), dim=3), dim=3), dim=3)

            feature1 = feature1.reshape([batchSize, int(x.size(1)/2 ), 12, 16] )
            feature1 = F.interpolate(feature1, scale_factor=2, mode='bilinear' )
            feature2 = feature2.reshape([batchSize, int(x.size(1)/2 ), 12, 16] )
            feature2 = F.interpolate(feature2, scale_factor=2, mode='bilinear' )

            combined = torch.cat([xPara, feature1/(8*8*8), feature2/(8*8*8)], dim=1 )

            featureOptim.zero_grad()
            normal1Opt, normal2Opt = decoder( combined )
            refraction, reflection, maskTr = renderer.forward(
                    originBatch, lookatBatch, upBatch,
                    envBatch,
                    normal1Opt, normal2Opt )
            renderedImgOpt = torch.clamp(refraction + reflection, 0, 1)

            maskPred = (1 - maskTr) * seg1Batch

            if opt.isMultiScale:
                error = []
                pre = renderedImgOpt
                gt = imBatch
                mask = maskPred
                for m in range(0, opt.scaleNum ):
                    pixelNum = torch.sum(mask )
                    error.append(torch.sum(torch.pow(pre - gt, 2) * mask ) / pixelNum )
                    pre = F.adaptive_avg_pool2d(pre, [int(pre.size(2)/2), int(pre.size(3)/2)] )
                    gt = F.adaptive_avg_pool2d(gt, [int(gt.size(2)/2), int(gt.size(3)/2)] )
                    mask = F.adaptive_avg_pool2d(mask, [int(mask.size(2)/2), int(mask.size(3)/2) ] )
                error = sum(error ) / opt.scaleNum
            else:
                pixel2Num = torch.sum(maskPred )
                error = torch.sum( (torch.pow(renderedImgOpt - imBatch, 2) * maskPred ) ) / pixel2Num

            pixel1Num = torch.sum(seg1Batch )

            errorTotal = error
            errorTotal.backward()

            featureOptim.step()

            if n % 20 == 0:
                print('%d/%d: Error %.3f' % (n, int(opt.iterationNum / 5), errorTotal.data.item() ) )
                if minError > errorTotal.data.item():
                    minError = errorTotal.data.item()
                else:
                    if n != 0:
                        break

        ##################### Get the final results #########################
        feature1 = torch.sum(torch.sum(torch.sum(cameraFeature1 * vhPara.unsqueeze(2), dim=3), dim=3), dim=3)
        feature2 = torch.sum(torch.sum(torch.sum(cameraFeature2 * vhPara.unsqueeze(2), dim=3), dim=3), dim=3)

        feature1 = feature1.reshape([batchSize, int(x.size(1)/2 ), 12, 16] )
        feature1 = F.interpolate(feature1, scale_factor=2, mode='bilinear' )
        feature2 = feature2.reshape([batchSize, int(x.size(1)/2 ), 12, 16] )
        feature2 = F.interpolate(feature2, scale_factor=2, mode='bilinear' )

        combined = torch.cat([xPara, feature1 / (8*8*8), feature2 / (8*8*8)], dim=1 )
        if iw > 0:
            normal1VHOpt, seg1VHOpt = decoderVH(feature1 )
            normal2VHOpt, seg2VHOpt = decoderVH(feature2 )

            normal1VHPreds.append(normal1VHOpt )
            seg1VHPreds.append(seg1VHOpt )
            normal2VHPreds.append(normal2VHOpt )
            seg2VHPreds.append(seg2VHOpt )

        normal1Opt, normal2Opt = decoder( combined )
        refraction, reflection, _ = renderer.forward(
                originBatch, lookatBatch, upBatch,
                envBatch,
                normal1Opt, normal2Opt )
        renderedImgOpt = torch.clamp(refraction + reflection, 0, 1)
        normal1Preds.append(normal1Opt )
        normal2Preds.append(normal2Opt )
        renderedImgs.append(renderedImgOpt )
    else:
        normal1Pred, normal2Pred = decoder( x )
        refraction, reflection, maskTr = renderer.forward(
                originBatch, lookatBatch, upBatch,
                envBatch,
                normal1Pred, normal2Pred )
        renderedImg = torch.clamp(refraction + reflection, 0, 1)

        maskPredOrigin = (1 - maskTr ) * seg1Batch

        # rescale real image
        scale = torch.sum(renderedImg*imBatch*maskPredOrigin) / torch.sum(renderedImg*renderedImg*maskPredOrigin)
        renderedImg = renderedImg * scale

        normal1Preds.append(normal1Pred )
        normal2Preds.append(normal2Pred )
        renderedImgs.append(renderedImg )
        maskPreds.append(maskPredOrigin)


        xPara = Variable(torch.FloatTensor(x.size() ) ).cuda()
        xPara.requires_grad = True
        xPara.data.copy_(x.data )
        featureOptim = optim.Adam([xPara], lr=1e-4, betas=(0.5, 0.999 ) )

        ########################### Optimize Normal 2 ###############################
        minError = 10
        for n in range(0, opt.iterationNum ):
            featureOptim.zero_grad()
            normal1Opt, normal2Opt = decoder( xPara )
            refraction, reflection, maskTr = renderer.forward(
                    originBatch, lookatBatch, upBatch,
                    envBatch,
                    normal1Opt, normal2Opt )
            renderedImgOpt = torch.clamp(refraction + reflection, 0, 1)

            maskPred = (1 - maskTr ) * seg1Batch

            # rescale real image
            scale = torch.sum(renderedImgOpt*imBatch*maskPred) / torch.sum(renderedImgOpt*renderedImgOpt*maskPred)
            renderedImgOpt = renderedImgOpt * scale

            if opt.isMultiScale:
                error = []
                pre = renderedImgOpt
                gt = imBatch
                mask = maskPred
                for m in range(0, opt.scaleNum ):
                    pixelNum = torch.sum(mask )
                    error.append(torch.sum(torch.pow(pre - gt, 2) * mask ) / pixelNum )
                    pre = F.adaptive_avg_pool2d(pre, [int(pre.size(2)/2), int(pre.size(3)/2)] )
                    gt = F.adaptive_avg_pool2d(gt, [int(gt.size(2)/2), int(gt.size(3)/2)] )
                    mask = F.adaptive_avg_pool2d(mask, [int(mask.size(2)/2), int(mask.size(3)/2) ] )
                error = sum(error ) / opt.scaleNum
            else:
                pixel2Num = torch.sum(maskPred )
                error = torch.sum( torch.pow(renderedImgOpt - imBatch, 2) * maskPred ) / pixel2Num

            pixel1Num = torch.sum( seg1Batch )
            errorNormal1 = torch.sum( torch.pow(normal1Opt - normal1Pred, 2) * seg1Batch ) / pixel1Num

            errorTotal = error + errorNormal1 * 10
            errorTotal.backward()
            featureOptim.step()

            if n % 20 == 0:
                print('%d/%d: Error %.3f' % (n, opt.iterationNum, errorTotal.data.item() ) )
                if minError > errorTotal.data.item():
                    minError = errorTotal.data.item()
                else:
                    if n != 0:
                        break

        #################### Optimize Normal 1 and 2 ###############################
        for n in range(0, int(opt.iterationNum / 5) ):
            featureOptim.zero_grad()
            normal1Opt, normal2Opt = decoder( xPara )
            refraction, reflection, maskTr = renderer.forward(
                    originBatch, lookatBatch, upBatch,
                    envBatch,
                    normal1Opt, normal2Opt )
            renderedImgOpt = torch.clamp(refraction + reflection, 0, 1)

            maskPred = (1 - maskTr ) * seg1Batch

            # rescale real image
            scale = torch.sum(renderedImgOpt*imBatch*maskPred) / torch.sum(renderedImgOpt*renderedImgOpt*maskPred)
            renderedImgOpt = renderedImgOpt * scale

            if opt.isMultiScale:
                error = []
                pre = renderedImgOpt
                gt = imBatch
                mask = maskPred
                for m in range(0, opt.scaleNum ):
                    pixelNum = torch.sum(mask )
                    error.append(torch.sum(torch.pow(pre - gt, 2) * mask ) / pixelNum )
                    pre = F.adaptive_avg_pool2d(pre, [int(pre.size(2)/2), int(pre.size(3)/2)] )
                    gt = F.adaptive_avg_pool2d(gt, [int(gt.size(2)/2), int(gt.size(3)/2)] )
                    mask = F.adaptive_avg_pool2d(mask, [int(mask.size(2)/2), int(mask.size(3)/2) ] )
                error = sum(error ) / opt.scaleNum
            else:
                pixel2Num = torch.sum(maskPred )
                error = torch.sum( torch.pow(renderedImgOpt - imBatch, 2) * maskPred ) / pixel2Num

            errorTotal = error
            errorTotal.backward()
            featureOptim.step()

            if n % 20 == 0:
                print('%d/%d: Error %.3f' % (n, int(opt.iterationNum / 5), errorTotal.data.item() ) )
                if minError > errorTotal.data.item():
                    minError = errorTotal.data.item()
                else:
                    if n != 0:
                        break

        ################## Get the final results ####################################
        normal1Opt, normal2Opt = decoder( xPara )
        refraction, reflection, maskTr = renderer.forward(
                originBatch, lookatBatch, upBatch,
                envBatch,
                normal1Opt, normal2Opt )
        renderedImgOpt = torch.clamp(refraction + reflection, 0, 1)

        maskPred = (1 - maskTr ) * seg1Batch

        # rescale real image
        scale = torch.sum(renderedImgOpt*imBatch*maskPred) / torch.sum(renderedImgOpt*renderedImgOpt*maskPred)
        renderedImgOpt = renderedImgOpt * scale

        normal1Preds.append(normal1Opt )
        normal2Preds.append(normal2Opt )
        renderedImgs.append(renderedImgOpt )
        maskPredFinal = (1 - maskTr ) * seg1Batch
        maskPreds.append(maskPredFinal)

    ########################################################

    end = time.time()
    '''
    # Compute the error
    normal1Errs = []
    meanAngle1Errs = []
    medianAngle1Errs = []

    normal2Errs = []
    meanAngle2Errs = []
    medianAngle2Errs = []

    renderedErrs = []

    if opt.isAddVisualHull and iw > 0:
        normal1VHErrs = []
        seg1VHErrs = []
        normal2VHErrs = []
        seg2VHErrs = []

    pixel1Num = max( (torch.sum(seg1Batch ).cpu().data).item(), 1)
    pixel2Num = max( (torch.sum(seg2Batch).cpu().data ).item(), 1 )
    for m in range(0, len(normal1Preds) ):
        normal1Errs.append( torch.sum( (normal1Preds[m] - normal1Batch)
                * (normal1Preds[m] - normal1Batch) * seg1Batch.expand_as(imBatch) ) / pixel1Num / 3.0 )
        meanAngle1Errs.append(180.0/np.pi * torch.sum(torch.acos(
            torch.clamp( torch.sum(normal1Preds[m] * normal1Batch, dim=1 ), -1, 1) )* seg1Batch.squeeze(1) ) / pixel1Num )
        if pixel1Num != 1:
            medianAngle1Errs.append(180.0/np.pi * torch.median(torch.acos(
                torch.clamp(torch.sum( normal1Preds[m] * normal1Batch, dim=1), -1, 1)[seg1Batch.squeeze(1) != 0] ) ) )
        else:
            medianAngle1Errs.append(0*torch.sum(normal1Preds[m]) )

    for m in range(0, len(normal2Preds) ):
        normal2Errs.append( torch.sum( (normal2Preds[m] - normal2Batch)
                * (normal2Preds[m] - normal2Batch) * seg1Batch.expand_as(imBatch) ) / pixel1Num / 3.0 )
        meanAngle2Errs.append(180.0/np.pi * torch.sum(torch.acos(
            torch.clamp( torch.sum(normal2Preds[m] * normal2Batch, dim=1 ), -1, 1) )* seg1Batch.squeeze(1) ) / pixel1Num )
        if pixel1Num != 1:
            medianAngle2Errs.append(180.0/np.pi * torch.median(torch.acos(
                torch.clamp(torch.sum( normal2Preds[m] * normal2Batch, dim=1), -1, 1)[seg1Batch.squeeze(1) != 0] ) ) )
        else:
            medianAngle2Errs.append(0*torch.sum(normal2Preds[m] ) )

    for m in range(0, len(renderedImgs ) ):
        renderedErrs.append( torch.sum( (renderedImgs[m] - imBgBatch)
            * (renderedImgs[m] - imBgBatch ) * seg2Batch.expand_as(imBatch ) ) / pixel2Num / 3.0 )

    if opt.isAddVisualHull and iw > 0:
        normal1Gt = F.adaptive_avg_pool2d(normal1Batch, (normal1VHPred.size(2), normal1VHPred.size(3) ) )
        normal2Gt = F.adaptive_avg_pool2d(normal2Batch, (normal2VHPred.size(2), normal2VHPred.size(3) ) )
        segGt = F.adaptive_avg_pool2d(seg1Batch, (seg1VHPred.size(2), seg1VHPred.size(3) ) )
        pixelVHNum = max(torch.sum(segGt ).cpu().data.item(), 1)


        for n in range(0, len(normal1VHPreds) ):
            normal1VHErrs.append(torch.sum( (normal1VHPreds[n] - normal1Gt)
                    * (normal1VHPreds[n] - normal1Gt) * segGt / pixelVHNum / 3.0 ) )
        for n in range(0, len(normal2VHPreds) ):
            normal2VHErrs.append(torch.sum( (normal2VHPreds[n] - normal2Gt)
                    * (normal2VHPreds[n] - normal2Gt) * segGt / pixelVHNum / 3.0 ) )
        for n in range(0, len(seg1VHPreds ) ):
            seg1VHErrs.append(torch.sum( (seg1VHPreds[n] - segGt ) * (seg1VHPreds[n] - segGt ) / pixelVHNum ) )
        for n in range(0, len(seg2VHPreds ) ):
            seg2VHErrs.append(torch.sum( (seg2VHPreds[n] - segGt ) * (seg2VHPreds[n] - segGt ) / pixelVHNum ) )

    # Output training error
    utils.writeErrToScreen('normal1', normal1Errs, epoch, j)
    utils.writeErrToScreen('medianAngle1', medianAngle1Errs, epoch, j)
    utils.writeErrToScreen('meanAngle1', meanAngle1Errs, epoch, j)
    utils.writeErrToScreen('normal2', normal2Errs, epoch, j)
    utils.writeErrToScreen('medianAngle2', medianAngle2Errs, epoch, j)
    utils.writeErrToScreen('meanAngle2', meanAngle2Errs, epoch, j)
    utils.writeErrToScreen('rendered', renderedErrs, epoch, j)

    if opt.isAddVisualHull and iw > 0:
        utils.writeErrToScreen('normal1VH', normal1VHErrs, epoch, j)
        utils.writeErrToScreen('normal2VH', normal2VHErrs, epoch, j)
        utils.writeErrToScreen('seg1VH', seg1VHErrs, epoch, j)
        utils.writeErrToScreen('seg2VH', seg2VHErrs, epoch, j)

    utils.writeErrToFile('normal1', normal1Errs, trainingLog, epoch, j)
    utils.writeErrToFile('medianAngle1', medianAngle1Errs, trainingLog, epoch, j)
    utils.writeErrToFile('meanAngle1', meanAngle1Errs, trainingLog, epoch, j)
    utils.writeErrToFile('normal2', normal2Errs, trainingLog, epoch, j)
    utils.writeErrToFile('medianAngle2', medianAngle2Errs, trainingLog, epoch, j)
    utils.writeErrToFile('meanAngle2', meanAngle2Errs, trainingLog, epoch, j)
    utils.writeErrToFile('rendered', renderedErrs, trainingLog, epoch, j)

    if opt.isAddVisualHull and iw > 0:
        utils.writeErrToFile('normal1VH', normal1VHErrs, trainingLog, epoch, j)
        utils.writeErrToFile('normal2VH', normal2VHErrs, trainingLog, epoch, j)
        utils.writeErrToFile('seg1VH', seg1VHErrs, trainingLog, epoch, j)
        utils.writeErrToFile('seg2VH', seg2VHErrs, trainingLog, epoch, j)

    normal1ErrsNpList = np.concatenate( [normal1ErrsNpList, utils.turnErrorIntoNumpy(normal1Errs)], axis=0 )
    medianAngle1ErrsNpList = np.concatenate( [medianAngle1ErrsNpList, utils.turnErrorIntoNumpy(medianAngle1Errs)], axis=0 )
    meanAngle1ErrsNpList = np.concatenate( [meanAngle1ErrsNpList, utils.turnErrorIntoNumpy(meanAngle1Errs)], axis=0 )
    normal2ErrsNpList = np.concatenate( [normal2ErrsNpList, utils.turnErrorIntoNumpy(normal2Errs)], axis=0 )
    medianAngle2ErrsNpList = np.concatenate( [medianAngle2ErrsNpList, utils.turnErrorIntoNumpy(medianAngle2Errs)], axis=0 )
    meanAngle2ErrsNpList = np.concatenate( [meanAngle2ErrsNpList, utils.turnErrorIntoNumpy(meanAngle2Errs)], axis=0 )
    renderedErrsNpList = np.concatenate( [renderedErrsNpList, utils.turnErrorIntoNumpy(renderedErrs ) ], axis=0 )

    if opt.isAddVisualHull and iw > 0:
        normal1VHErrsNpList = np.concatenate([normal1VHErrsNpList, utils.turnErrorIntoNumpy(normal1VHErrs )  ], axis=0 )
        seg1VHErrsNpList = np.concatenate([seg1VHErrsNpList, utils.turnErrorIntoNumpy(seg1VHErrs ) ], axis=0 )
        normal2VHErrsNpList = np.concatenate([normal2VHErrsNpList, utils.turnErrorIntoNumpy(normal2VHErrs ) ], axis=0 )
        seg2VHErrsNpList = np.concatenate([seg2VHErrsNpList, utils.turnErrorIntoNumpy(seg2VHErrs ) ], axis=0 )

    utils.writeNpErrToScreen('normal1Accu', np.mean(normal1ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('medianAngle1Accu', np.mean(medianAngle1ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('meanAngle1Accu', np.mean(meanAngle1ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('normal2Accu', np.mean(normal2ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('medianAngle2Accu', np.mean(medianAngle2ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('meanAngle2Accu', np.mean(meanAngle2ErrsNpList[1:j+1, :], axis=0), epoch, j)
    utils.writeNpErrToScreen('renderedAccu', np.mean(renderedErrsNpList[1:j+1, :], axis=0), epoch, j)

    if opt.isAddVisualHull and iw > 0:
        utils.writeNpErrToScreen('normal1VHAccu', np.mean(normal1VHErrsNpList[1:j+1, :], axis=0), epoch, j)
        utils.writeNpErrToScreen('normal2VHAccu', np.mean(normal2VHErrsNpList[1:j+1, :], axis=0), epoch, j)
        utils.writeNpErrToScreen('seg1VHAccu', np.mean(seg1VHErrsNpList[1:j+1, :], axis=0), epoch, j)
        utils.writeNpErrToScreen('seg2VHAccu', np.mean(seg2VHErrsNpList[1:j+1, :], axis=0), epoch, j)

    utils.writeNpErrToFile('normal1Accu', np.mean(normal1ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    utils.writeNpErrToFile('medianAngle1Accu', np.mean(medianAngle1ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    utils.writeNpErrToFile('meanAngle1Accu', np.mean(meanAngle1ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    utils.writeNpErrToFile('normal2Accu', np.mean(normal2ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    utils.writeNpErrToFile('medianAngle2Accu', np.mean(medianAngle2ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    utils.writeNpErrToFile('meanAngle2Accu', np.mean(meanAngle2ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    utils.writeNpErrToFile('renderedAccu', np.mean(renderedErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)

    if opt.isAddVisualHull and iw > 0:
        utils.writeNpErrToFile('normal1VHAccu', np.mean(normal1VHErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
        utils.writeNpErrToFile('normal2VHAccu', np.mean(normal2VHErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
        utils.writeNpErrToFile('seg1VHAccu', np.mean(seg1VHErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
        utils.writeNpErrToFile('seg2VHAccu', np.mean(seg2VHErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    '''
    #if j==1 or j% 50 == 0:
    if True:
        if opt.batchSize == 1:
            path = nameBatch[0][0].replace(opt.dataRoot+'/', '').replace('.png', '')
            p = os.path.dirname(os.path.join(opt.optimRoot, path))
            if os.path.exists(p) == False:
                os.makedirs(p)
        else:
            path = '{0}_{1}'.format(j, shapeName)
        '''
        vutils.save_image( (0.5*(normal1Batch + 1)*seg1Batch.expand_as(imBatch) ).data,
                '{0}/{1}_normal1Gt.png'.format(opt.optimRoot, j), nrow=5 )
        vutils.save_image( (0.5*(normal2Batch + 1)*seg1Batch.expand_as(imBatch) ).data,
                '{0}/{1}_normal2Gt.png'.format(opt.optimRoot, j), nrow=5 )

        vutils.save_image( (torch.clamp(imBatch, 0, 1)**(1.0/2.2) *seg2Batch.expand_as(imBatch) ).data,
                '{0}/{1}_im.png'.format(opt.optimRoot, j), nrow=5 )
        vutils.save_image( ( torch.clamp(imBgBatch, 0, 1)**(1.0/2.2) ).data,
                '{0}/{1}_imBg.png'.format(opt.optimRoot, j), nrow=5 )

        vutils.save_image( seg2Batch,
                '{0}/{1}_mask2.png'.format(opt.optimRoot, j), nrow=5 )
        '''
        # Save real rgb images with segmentation mask
        vutils.save_image( (torch.clamp(imBatch, 0, 1)*seg1Batch.expand_as(imBatch) ).data,
                '{0}/{1}_im.png'.format(opt.optimRoot, path), nrow=5 )
        vutils.save_image( ( torch.clamp(imBgBatch, 0, 1)).data,
                '{0}/{1}_imBg.png'.format(opt.optimRoot, path), nrow=5 )

        # Save the predicted results
        for n in range(0, len(normal1Preds) ):
            vutils.save_image( ( 0.5*(normal1Preds[n] + 1)*seg1Batch.expand_as(imBatch) ).data,
                    '{0}/{1}_normal1Pred_{2}.png'.format(opt.optimRoot, path, n), nrow=5 )
        for n in range(0, len(normal2Preds) ):
            vutils.save_image( ( 0.5*(normal2Preds[n] + 1)*seg1Batch.expand_as(imBatch) ).data,
                    '{0}/{1}_normal2Pred_{2}.png'.format(opt.optimRoot, path, n), nrow=5 )
        for n in range(0, len(renderedImgs ) ):
            vutils.save_image( (torch.clamp(renderedImgs[n], 0, 1) *maskPreds[n].expand_as(imBatch) ) .data,
                    '{0}/{1}_renederedImg_{2}.png'.format(opt.optimRoot, path, n), nrow=5 )
        '''
        if opt.isAddVisualHull:
            utils.writePointCloud('{0}/{1}_mesh.ply'.format(opt.optimRoot, j), visualHull, colormap )
            utils.writeVisualHull('{0}/{1}_pointCloud.ply'.format(opt.optimRoot, j), visualHull )
            if iw > 0:
                for n in range(0, len(normal1VHPreds ) ):
                    vutils.save_image( (0.5*(normal1VHPreds[n] + 1)*segGt ).data,
                            '{0}/{1}_normal1VHPred_{2}.png'.format(opt.optimRoot, j, n), nrow=5 )
                for n in range(0, len(normal2VHPreds ) ):
                    vutils.save_image( (0.5*(normal2VHPreds[n] + 1)*segGt ).data,
                            '{0}/{1}_normal2VHPred_{2}.png'.format(opt.optimRoot, j, n), nrow=5 )
                for n in range(0, len(normal1VHPreds ) ):
                    vutils.save_image( seg1VHPreds[n].data,
                            '{0}/{1}_seg1VHPred_{2}.png'.format(opt.optimRoot, j, n), nrow=5 )
                for n in range(0, len(seg2VHPreds ) ):
                    vutils.save_image( seg2VHPreds[n].data,
                            '{0}/{1}_seg2VHPred_{2}.png'.format(opt.optimRoot, j, n), nrow=5 )

                vutils.save_image( (0.5*(normal1Gt + 1)*segGt ).data,
                        '{0}/{1}_normal1VHGt.png'.format(opt.optimRoot, j), nrow=5 )
                vutils.save_image( (0.5*(normal2Gt + 1)*segGt ).data,
                        '{0}/{1}_normal2VHGt.png'.format(opt.optimRoot, j), nrow=5 )
        '''
    # Output the normal direction
    for n in range(0, batchSize ):
        normal1 = normal1Preds[2][n, :].cpu().data.numpy().transpose([1, 2, 0] )
        normal2 = normal2Preds[2][n, :].cpu().data.numpy().transpose([1, 2, 0] )
        normal1 = normal1 * seg1Batch[n, :].cpu().data.numpy().transpose([1, 2, 0] )
        normal2 = normal2 * seg1Batch[n, :].cpu().data.numpy().transpose([1, 2, 0] )
        twoNormals = np.concatenate([normal1, normal2], axis=2 )

        if opt.isH5py:
            #name = nameBatch[n][0].replace('im_', 'imtwoNormalPred%d_' % opt.camNum ).replace('.rgbe', '.h5')
            name = nameBatch[n][0].replace('im_', 'imtwoNormalPred%d_' % viewNum ).replace('.png', '.h5')
            hf = h5py.File(name, 'w')
            hf.create_dataset('data', data = twoNormals, compression='lzf')
            hf.close()
        else:
            #name = nameBatch[n][0].replace('im_', 'imtwoNormalPred%d_' % opt.camNum ).replace('.rgbe', '.npy')
            name = nameBatch[n][0].replace('im_', 'imtwoNormalPred%d_' % viewNum ).replace('.png', '.npy')
            np.save(name, twoNormals )
        print('Save normal %s' % name )

    torch.cuda.empty_cache()

    timeSum += (end - start )
    countSum += 1

    break


trainingLog.close()
print(timeSum / countSum )
'''
# Save the error record
np.save('{0}/normal1Error_{1}.npy'.format(opt.optimRoot, epoch), normal1ErrsNpList )
np.save('{0}/medianAngle1Error_{1}.npy'.format(opt.optimRoot, epoch), medianAngle1ErrsNpList )
np.save('{0}/meanAngle1Error_{1}.npy'.format(opt.optimRoot, epoch), meanAngle1ErrsNpList )
np.save('{0}/normal2Error_{1}.npy'.format(opt.optimRoot, epoch), normal2ErrsNpList )
np.save('{0}/medianAngle2Error_{1}.npy'.format(opt.optimRoot, epoch), medianAngle2ErrsNpList )
np.save('{0}/meanAngle2Error_{1}.npy'.format(opt.optimRoot, epoch), meanAngle2ErrsNpList )
np.save('{0}/renderError_{1}.npy'.format(opt.optimRoot, epoch), renderedErrsNpList )

if opt.isAddVisualHull and iw > 0.0:
    np.save('{0}/normal1VHError_{1}.npy'.format(opt.optimRoot, epoch), normal1VHErrsNpList )
    np.save('{0}/seg1VHError_{1}.npy'.format(opt.optimRoot, epoch), seg1VHErrsNpList )
    np.save('{0}/normal2VHError_{1}.npy'.format(opt.optimRoot, epoch), normal2VHErrsNpList )
    np.save('{0}/seg2VHError_{1}.npy'.format(opt.optimRoot, epoch), seg2VHErrsNpList )
'''

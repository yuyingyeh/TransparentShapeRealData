import torch
import numpy as np
from torch.autograd import Variable
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

import time

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='../../createRealData/ImagesReal/real/', help='path to images' )
parser.add_argument('--shapeRoot', default='../../createRealData/Shapes/real/', help='path to images' )
parser.add_argument('--experiment', default=None, help='the path to store samples and models' )
parser.add_argument('--testRoot', default=None, help='the path to store outputs')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=10, help='the number of epochs for training' )
parser.add_argument('--batchSize', type=int, default=1, help='input batch size' )
parser.add_argument('--imageHeight', type=int, default=192, help='the height / width of the input image to network' )
parser.add_argument('--imageWidth', type=int, default=256, help='the height / width of the input image to network' )
parser.add_argument('--envHeight', type=int, default=256, help='the height / width of the input envmap to network' )
parser.add_argument('--envWidth', type=int, default=512, help='the height / width of the input envmap to network' )
parser.add_argument('--volumeSize', type=int, default=64, help='the size of the volume' )
# The parameters
parser.add_argument('--camNum', type=int, default=10, help='the number of views to create the visual hull' )
parser.add_argument('--camNumReal', type=int, default=16, help='the number of views to create the real visual hull' )
parser.add_argument('--sampleNum', type=int, default=1, help='the sample num for the cost volume')
parser.add_argument('--shapeStart', type=int, default=0, help='the start id of the shape')
parser.add_argument('--shapeEnd', type=int, default=600, help='the end id of the shape')
parser.add_argument('--isAddCostVolume', action='store_true', help='whether to use cost volume or not' )
parser.add_argument('--poolingMode', type=int, default=2, help='0: maxpooling, 1: average pooling 2: learnable pooling' )
parser.add_argument('--isAddVisualHull', action='store_true', help='whether to build visual hull into training' )
parser.add_argument('--isNoErrMap', action='store_true', help = 'whether to remove the error map in the input')
# The rendering parameters
parser.add_argument('--eta1', type=float, default=1.0003, help='the index of refraction of air' )
parser.add_argument('--eta2', type=float, default=1.4723, help='the index of refraction of glass' )
parser.add_argument('--fov', type=float, default=63.35, help='the field of view of camera' )
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
#opt.dataRoot = opt.dataRoot % opt.camNum

nw = opt.normalWeight
rw = opt.renderWeight
iw = opt.intermediateWeight

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
    if opt.isNoErrMap:
        opt.experiment += '_noerr'
    if opt.isAddCostVolume:
        if opt.poolingMode == 0:
            opt.experiment +=  '_volume_sp%d_an%d_maxpool' % (opt.sampleNum, angleNum )
        elif opt.poolingMode == 1:
            opt.experiment += '_volume_sp%d_an%d_avgpool' % (opt.sampleNum, angleNum )
        elif opt.poolingMode == 2:
            opt.experiment += '_volume_sp%d_an%d_weigtedSum' % (opt.sampleNum, angleNum )
        else:
            print("Wrong: unrecognizable pooling mode.")
            assert(False)
    if opt.isAddVisualHull:
        opt.experiment += '_vh_iw%.2f' % iw

if not osp.isdir(opt.experiment ):
    print('Warning: the model %s does not exist' % opt.experiment )
    assert(False )
opt.testRoot = opt.experiment.replace('check', 'test')

####################
opt.testRoot = 'temp_countTime'
####################

os.system('mkdir {0}'.format( opt.testRoot ) )
os.system('cp *.py %s' % opt.testRoot )

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
'''
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
'''
##############  ######################
# Send things into GPU
if opt.cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    normalFeature = normalFeature.cuda()
    normalPool = normalPool.cuda()
    '''
    if opt.isAddVisualHull:
        encoderVH = encoderVH.cuda()
        encoderCamera = encoderCamera.cuda()
        decoderVH = decoderVH.cuda()
    '''
####################################

# Other modules
renderer = models.renderer(eta1 = opt.eta1, eta2 = opt.eta2,
        isCuda = opt.cuda, gpuId = opt.gpuId,
        batchSize = opt.batchSize,
        fov = opt.fov,
        imWidth=opt.imageWidth, imHeight = opt.imageHeight,
        envWidth = opt.envWidth, envHeight = opt.envHeight )

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
        isRandom = False, phase='TEST', rseed = 1,
        isLoadVH = True, isLoadEnvmap = True, isLoadCam = True,
        shapeRs = opt.shapeStart, shapeRe = opt.shapeEnd,
        camNum = opt.camNumReal, batchSize = opt.batchSize )
brdfLoader = DataLoader( brdfDataset, batch_size = 1, num_workers = 0, shuffle = False )

j = 0
'''
normal1ErrsNpList = np.ones( [1, 2], dtype = np.float32 )
meanAngle1ErrsNpList = np.ones([1, 2], dtype = np.float32 )
medianAngle1ErrsNpList = np.ones([1, 2], dtype = np.float32 )
normal2ErrsNpList = np.ones( [1, 2], dtype = np.float32 )
meanAngle2ErrsNpList = np.ones([1, 2], dtype = np.float32 )
medianAngle2ErrsNpList = np.ones([1, 2], dtype = np.float32 )
renderedErrsNpList = np.ones([1, 2], dtype=np.float32 )

if opt.isAddVisualHull and iw > 0.0:
        normal1VHErrsNpList = np.ones( [1, 1], dtype = np.float32 )
        seg1VHErrsNpList = np.ones([1, 1], dtype=np.float32 )
        normal2VHErrsNpList = np.ones( [1, 1], dtype = np.float32 )
        seg2VHErrsNpList = np.ones([1, 1], dtype=np.float32 )
'''
epoch = opt.nepoch
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.testRoot, epoch ), 'w' )
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
    #seg1_cpu = dataBatch['seg1']
    seg1Batch = Variable((seg1_cpu ) ).cuda()
    '''
    normal2_cpu = dataBatch['normal2'].squeeze(0 )
    normal2Batch = Variable(normal2_cpu ).cuda()

    seg2_cpu = dataBatch['seg2'].squeeze(0 )
    seg2Batch = Variable(seg2_cpu ).cuda()
    '''
    # Load the image from cpu to gpu
    im_cpu = dataBatch['im'].squeeze(0 )
    #im_cpu = dataBatch['im']
    imBatch = Variable(im_cpu ).cuda()

    imBg_cpu = dataBatch['imE'].squeeze(0 )
    #imBg_cpu = dataBatch['imE']
    imBgBatch = Variable(imBg_cpu ).cuda()

    # Load environment map
    envmap_cpu = dataBatch['env'].squeeze(0 )
    #envmap_cpu = dataBatch['env']
    envBatch = Variable(envmap_cpu ).cuda()

    # Load camera parameters
    origin_cpu = dataBatch['origin'].squeeze(0 )
    #origin_cpu = dataBatch['origin']
    originBatch = Variable(origin_cpu ).cuda()

    lookat_cpu = dataBatch['lookat'].squeeze(0 )
    #lookat_cpu = dataBatch['lookat']
    lookatBatch = Variable(lookat_cpu ).cuda()

    up_cpu = dataBatch['up'].squeeze(0 )
    #up_cpu = dataBatch['up']
    upBatch = Variable(up_cpu ).cuda()

    # Load visual hull data
    normal1VH_cpu = dataBatch['normal1VH'].squeeze(0)
    #normal1VH_cpu = dataBatch['normal1VH']
    normal1VHBatch = Variable(normal1VH_cpu ).cuda()

    seg1VH_cpu = dataBatch['seg1VH'].squeeze(0 )
    #seg1VH_cpu = dataBatch['seg1VH']
    seg1VHBatch = Variable((seg1VH_cpu ) ).cuda()

    normal2VH_cpu = dataBatch['normal2VH'].squeeze(0 )
    #normal2VH_cpu = dataBatch['normal2VH']
    normal2VHBatch = Variable(normal2VH_cpu ).cuda()

    seg2VH_cpu = dataBatch['seg2VH'].squeeze(0 )
    #seg2VH_cpu = dataBatch['seg2VH']
    seg2VHBatch = Variable(seg2VH_cpu ).cuda()

    nameBatch = dataBatch['name']

    batchSize = normal1VHBatch.size(0 )
    ########################################################
    normal1Preds = []
    normal2Preds = []
    renderedImgs = []
    maskPreds = []


    #### start counding time #
    start = time.time()
    ##########################

    refraction, reflection, maskVH = renderer.forward(
            originBatch, lookatBatch, upBatch,
            envBatch,
            normal1VHBatch, normal2VHBatch )
    renderedImg = torch.clamp(refraction + reflection, 0, 1 )

    # rescale real image
    scale = torch.sum(renderedImg*imBatch*(1-maskVH)) / torch.sum(renderedImg*renderedImg*(1-maskVH))
    renderedImg = renderedImg * scale

    normal1Preds.append(normal1VHBatch )
    normal2Preds.append(normal2VHBatch )
    renderedImgs.append(renderedImg )
    maskPreds.append(1-maskVH)
    errorVH = torch.sum(torch.pow(renderedImg - imBatch, 2.0) * seg1Batch, dim=1).unsqueeze(1)

    if opt.isNoErrMap:
        inputBatch = torch.cat([imBatch, imBgBatch, seg1Batch,
            normal1VHBatch, normal2VHBatch, errorVH * 0, maskVH * 0], dim=1 )
    else:
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

    '''
    if opt.isAddVisualHull:
        error = torch.sum(torch.pow(renderedImgs[0] - imBatch, 2), dim=1).unsqueeze(1)
        error = torch.exp(-2 * error ) * seg1Batch
        error = F.adaptive_avg_pool2d(error, [int(0.75*opt.volumeSize ), opt.volumeSize ] )
        mask = F.adaptive_avg_pool2d(seg1Batch, [int(0.75*opt.volumeSize ), opt.volumeSize ] )
        visualHull = buildVisualHull.forward(originBatch, lookatBatch, upBatch, error, mask )
        vhFeature = encoderVH(visualHull )
        vhFeature = vhFeature.unsqueeze(2 )

        yAxis = upBatch / torch.sqrt(torch.sum(upBatch * upBatch, dim=1) ).unsqueeze(1)
        zAxis = lookatBatch - originBatch
        zAxis = zAxis / torch.sqrt(torch.sum(zAxis * zAxis, dim=1) ).unsqueeze(1)
        xAxis = torch.cross(zAxis, yAxis, dim=1 )
        cameraFeature = encoderCamera(torch.cat([xAxis, yAxis, zAxis, originBatch], dim=1) )
        cameraFeature = cameraFeature.reshape(batchSize, 2, 16 * 12, 8, 8, 8)
        cameraFeature1, cameraFeature2 = torch.split(cameraFeature, [1, 1], dim=1 )

        feature1 = torch.sum(torch.sum(torch.sum(cameraFeature1 * vhFeature, dim=3), dim=3), dim=3)
        feature2 = torch.sum(torch.sum(torch.sum(cameraFeature2 * vhFeature, dim=3), dim=3), dim=3)

        feature1 = feature1.reshape([batchSize, 256, 12, 16] )
        feature1 = F.interpolate(feature1, scale_factor=2, mode='bilinear' )
        feature2 = feature2.reshape([batchSize, 256, 12, 16] )
        feature2 = F.interpolate(feature2, scale_factor=2, mode='bilinear' )

        x = torch.cat([x, feature1/(8*8*8), feature2/(8*8*8) ], dim=1 )

        if iw > 0:
            normal1VHPred, seg1VHPred = decoderVH(feature1 )
            normal2VHPred, seg2VHPred = decoderVH(feature2 )
    '''
    normal1Pred, normal2Pred = decoder(x )
    refraction, reflection, maskTr = renderer.forward(
            originBatch, lookatBatch, upBatch,
            envBatch,
            normal1Pred, normal2Pred )
    renderedImg = torch.clamp(refraction + reflection, 0, 1)

    maskPred = (1-maskTr)*seg1Batch
    # rescale real image
    scale = torch.sum(renderedImg*imBatch*maskPred) / torch.sum(renderedImg*renderedImg*maskPred)
    renderedImg = renderedImg * scale

    normal1Preds.append(normal1Pred )
    normal2Preds.append(normal2Pred )
    renderedImgs.append(renderedImg )
    maskPreds.append((1-maskTr)*seg1Batch)
    ########################################################

    #######################
    end = time.time()
    ########################

    timeSum += (end - start )
    countSum += 1

    '''
    # Compute the error
    normal1Errs = []
    meanAngle1Errs = []
    medianAngle1Errs = []

    normal2Errs = []
    meanAngle2Errs = []
    medianAngle2Errs = []

    renderedErrs = []

    pixel1Num = max( (torch.sum(seg1Batch ).cpu().data).item(), 1 )
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

        normal1VHErr = torch.sum( (normal1VHPred - normal1Gt)
                * (normal1VHPred - normal1Gt) * segGt / pixelVHNum / 3.0 )
        normal2VHErr = torch.sum( (normal2VHPred - normal2Gt)
                * (normal2VHPred - normal2Gt) * segGt / pixelVHNum / 3.0 )
        seg1VHErr = torch.sum( (seg1VHPred - segGt ) * (seg1VHPred - segGt ) / pixelVHNum )
        seg2VHErr = torch.sum( (seg2VHPred - segGt ) * (seg2VHPred - segGt ) / pixelVHNum )

        intermediateError = normal1VHErr + normal2VHErr + seg1VHErr + seg2VHErr


    # Output training error
    utils.writeErrToScreen('normal1', normal1Errs, epoch, j)
    utils.writeErrToScreen('medianAngle1', medianAngle1Errs, epoch, j)
    utils.writeErrToScreen('meanAngle1', meanAngle1Errs, epoch, j)
    utils.writeErrToScreen('normal2', normal2Errs, epoch, j)
    utils.writeErrToScreen('medianAngle2', medianAngle2Errs, epoch, j)
    utils.writeErrToScreen('meanAngle2', meanAngle2Errs, epoch, j)
    utils.writeErrToScreen('rendered', renderedErrs, epoch, j)

    if opt.isAddVisualHull and iw > 0:
        utils.writeErrToScreen('normal1VH', [normal1VHErr ], epoch, j)
        utils.writeErrToScreen('normal2VH', [normal2VHErr ], epoch, j)
        utils.writeErrToScreen('seg1VH', [seg1VHErr ], epoch, j)
        utils.writeErrToScreen('seg2VH', [seg2VHErr ], epoch, j)

    utils.writeErrToFile('normal1', normal1Errs, testingLog, epoch, j)
    utils.writeErrToFile('medianAngle1', medianAngle1Errs, testingLog, epoch, j)
    utils.writeErrToFile('meanAngle1', meanAngle1Errs, testingLog, epoch, j)
    utils.writeErrToFile('normal2', normal2Errs, testingLog, epoch, j)
    utils.writeErrToFile('medianAngle2', medianAngle2Errs, testingLog, epoch, j)
    utils.writeErrToFile('meanAngle2', meanAngle2Errs, testingLog, epoch, j)
    utils.writeErrToFile('rendered', renderedErrs, testingLog, epoch, j)

    if opt.isAddVisualHull and iw > 0:
        utils.writeErrToFile('normal1VH', [normal1VHErr ], testingLog, epoch, j)
        utils.writeErrToFile('normal2VH', [normal2VHErr ], testingLog, epoch, j)
        utils.writeErrToFile('seg1VH', [seg1VHErr ], testingLog, epoch, j)
        utils.writeErrToFile('seg2VH', [seg2VHErr ], testingLog, epoch, j)

    normal1ErrsNpList = np.concatenate( [normal1ErrsNpList, utils.turnErrorIntoNumpy(normal1Errs)], axis=0 )
    medianAngle1ErrsNpList = np.concatenate( [medianAngle1ErrsNpList, utils.turnErrorIntoNumpy(medianAngle1Errs)], axis=0 )
    meanAngle1ErrsNpList = np.concatenate( [meanAngle1ErrsNpList, utils.turnErrorIntoNumpy(meanAngle1Errs)], axis=0 )
    normal2ErrsNpList = np.concatenate( [normal2ErrsNpList, utils.turnErrorIntoNumpy(normal2Errs)], axis=0 )
    medianAngle2ErrsNpList = np.concatenate( [medianAngle2ErrsNpList, utils.turnErrorIntoNumpy(medianAngle2Errs)], axis=0 )
    meanAngle2ErrsNpList = np.concatenate( [meanAngle2ErrsNpList, utils.turnErrorIntoNumpy(meanAngle2Errs)], axis=0 )
    renderedErrsNpList = np.concatenate( [renderedErrsNpList, utils.turnErrorIntoNumpy(renderedErrs ) ], axis=0 )

    if opt.isAddVisualHull and iw > 0:
        normal1VHErrsNpList = np.concatenate([normal1VHErrsNpList, utils.turnErrorIntoNumpy([normal1VHErr ]) ], axis=0 )
        seg1VHErrsNpList = np.concatenate([seg1VHErrsNpList, utils.turnErrorIntoNumpy([seg1VHErr ]) ], axis=0 )
        normal2VHErrsNpList = np.concatenate([normal2VHErrsNpList, utils.turnErrorIntoNumpy([normal2VHErr ]) ], axis=0 )
        seg2VHErrsNpList = np.concatenate([seg2VHErrsNpList, utils.turnErrorIntoNumpy([seg2VHErr ]) ], axis=0 )

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

    utils.writeNpErrToFile('normal1Accu', np.mean(normal1ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('medianAngle1Accu', np.mean(medianAngle1ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('meanAngle1Accu', np.mean(meanAngle1ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('normal2Accu', np.mean(normal2ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('medianAngle2Accu', np.mean(medianAngle2ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('meanAngle2Accu', np.mean(meanAngle2ErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    utils.writeNpErrToFile('renderedAccu', np.mean(renderedErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)

    if opt.isAddVisualHull and iw > 0:
        utils.writeNpErrToFile('normal1VHAccu', np.mean(normal1VHErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
        utils.writeNpErrToFile('normal2VHAccu', np.mean(normal2VHErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
        utils.writeNpErrToFile('seg1VHAccu', np.mean(seg1VHErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
        utils.writeNpErrToFile('seg2VHAccu', np.mean(seg2VHErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j)
    '''
    #if j == 1 or j% 200 == 0:
    if True:
        if opt.batchSize == 1:
            path = nameBatch[0][0].replace(opt.dataRoot, '').replace('.png', '')
            p = os.path.dirname(os.path.join(opt.testRoot, path))
            if os.path.exists(p) == False:
                os.makedirs(p)
        else:
            path = j
        '''
        vutils.save_image( (0.5*(normal1Batch + 1)*seg1Batch.expand_as(imBatch) ).data,
                '{0}/{1}_normal1Gt.png'.format(opt.testRoot, j), nrow=5 )
        vutils.save_image( (0.5*(normal2Batch + 1)*seg1Batch.expand_as(imBatch) ).data,
                '{0}/{1}_normal2Gt.png'.format(opt.testRoot, j), nrow=5 )

        vutils.save_image( (torch.clamp(imBatch, 0, 1)**(1.0/2.2) *seg2Batch.expand_as(imBatch) ).data,
                '{0}/{1}_im.png'.format(opt.testRoot, j), nrow=5 )
        vutils.save_image( ( torch.clamp(imBgBatch, 0, 1)**(1.0/2.2) ).data,
                '{0}/{1}_imBg.png'.format(opt.testRoot, j), nrow=5 )

        vutils.save_image( seg2Batch,
                '{0}/{1}_mask2.png'.format(opt.testRoot, j), nrow=5 )
        '''
        # Save visual hull Initialization
        vutils.save_image( (0.5*(normal1VHBatch + 1)*seg1VHBatch.expand_as(imBatch) ).data,
                '{0}/{1}_normal1VH.png'.format(opt.testRoot, path), nrow=5 )
        vutils.save_image( (0.5*(normal2VHBatch + 1)*seg1VHBatch.expand_as(imBatch) ).data,
                '{0}/{1}_normal2VH.png'.format(opt.testRoot, path), nrow=5 )

        # Save real rgb images with segmentation mask
        vutils.save_image( (torch.clamp(imBatch, 0, 1)*seg1Batch.expand_as(imBatch) ).data,
                '{0}/{1}_im.png'.format(opt.testRoot, path), nrow=5 )
        vutils.save_image( ( torch.clamp(imBgBatch, 0, 1)).data,
                '{0}/{1}_imBg.png'.format(opt.testRoot, path), nrow=5 )


        # Save the predicted results
        for n in range(0, len(normal1Preds) ):
            vutils.save_image( ( 0.5*(normal1Preds[n] + 1)*seg1Batch.expand_as(imBatch) ).data,
                    '{0}/{1}_normal1Pred_{2}.png'.format(opt.testRoot, path, n), nrow=5 )
        for n in range(0, len(normal2Preds) ):
            vutils.save_image( ( 0.5*(normal2Preds[n] + 1)*seg1Batch.expand_as(imBatch) ).data,
                    '{0}/{1}_normal2Pred_{2}.png'.format(opt.testRoot, path, n), nrow=5 )
        for n in range(0, len(renderedImgs ) ):
            vutils.save_image( (torch.clamp(renderedImgs[n], 0, 1)*maskPreds[n] .expand_as(imBatch) ) .data,
                    '{0}/{1}_renederedImg_{2}.png'.format(opt.testRoot, path, n), nrow=5 )
        '''
        for n in range(0, len(renderedImgs ) ):
            vutils.save_image( (torch.clamp(renderedImgs[n], 0, 1)**(1.0/2.2) *seg2Batch.expand_as(imBatch) ) .data,
                    '{0}/{1}_renederedImg_{2}.png'.format(opt.testRoot, j, n), nrow=5 )
        '''
        '''
        if opt.isAddVisualHull:
            utils.writePointCloud('{0}/{1}_mesh.ply'.format(opt.testRoot, j), visualHull, colormap )
            utils.writeVisualHull('{0}/{1}_pointCloud.ply'.format(opt.testRoot, j), visualHull )
            if iw > 0:
                vutils.save_image( (0.5*(normal1VHPred + 1)*segGt ).data,
                        '{0}/{1}_normal1VHPred.png'.format(opt.testRoot, j), nrow=5 )
                vutils.save_image( (0.5*(normal2VHPred + 1)*segGt ).data,
                        '{0}/{1}_normal2VHPred.png'.format(opt.testRoot, j), nrow=5 )
                vutils.save_image( seg1VHPred.data,
                        '{0}/{1}_seg1VHPred.png'.format(opt.testRoot, j), nrow=5 )
                vutils.save_image( seg2VHPred.data,
                        '{0}/{1}_seg2VHPred.png'.format(opt.testRoot, j), nrow=5 )

                vutils.save_image( (0.5*(normal1Gt + 1)*segGt ).data,
                        '{0}/{1}_normal1VHGt.png'.format(opt.testRoot, j), nrow=5 )
                vutils.save_image( (0.5*(normal2Gt + 1)*segGt ).data,
                        '{0}/{1}_normal2VHGt.png'.format(opt.testRoot, j), nrow=5 )
        '''

testingLog.close()


print(timeSum / countSum )

'''
# Save the error record
np.save('{0}/normal1Error_{1}.npy'.format(opt.testRoot, epoch), normal1ErrsNpList )
np.save('{0}/medianAngle1Error_{1}.npy'.format(opt.testRoot, epoch), medianAngle1ErrsNpList )
np.save('{0}/meanAngle1Error_{1}.npy'.format(opt.testRoot, epoch), meanAngle1ErrsNpList )
np.save('{0}/normal2Error_{1}.npy'.format(opt.testRoot, epoch), normal2ErrsNpList )
np.save('{0}/medianAngle2Error_{1}.npy'.format(opt.testRoot, epoch), medianAngle2ErrsNpList )
np.save('{0}/meanAngle2Error_{1}.npy'.format(opt.testRoot, epoch), meanAngle2ErrsNpList )
np.save('{0}/renderError_{1}.npy'.format(opt.testRoot, epoch), renderedErrsNpList )

if opt.isAddVisualHull and iw > 0.0:
    np.save('{0}/normal1VHError_{1}.npy'.format(opt.testRoot, epoch), normal1VHErrsNpList )
    np.save('{0}/seg1VHError_{1}.npy'.format(opt.testRoot, epoch), seg1VHErrsNpList )
    np.save('{0}/normal2VHError_{1}.npy'.format(opt.testRoot, epoch), normal2VHErrsNpList )
    np.save('{0}/seg2VHError_{1}.npy'.format(opt.testRoot, epoch), seg2VHErrsNpList )
'''

import os.path as osp
import numpy as np
import glob
import struct
import cv2
import scipy.ndimage as ndimage
import open3d as o3d
import os
import argparse
import struct
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='real')
parser.add_argument('--renderProgram', default='/home/zhl/CVPR20/TransparentShape/OptixRenderer/src/bin/optixRenderer', help='path to the rendering program')
parser.add_argument('--fileRoot', default='./Shapes/', help='path to the file root')
parser.add_argument('--outputRoot', default='./Images', help='path to the output root')
parser.add_argument('--forceOutput', action='store_true', help='Overwrite previous results')
parser.add_argument('--shapeId', default=None, help='if not none, eval on selected shapeId')
#parser.add_argument('--rs', default = 0, type=int, help='the starting point')
#parser.add_argument('--re', default = 1000, type=int, help='the end point')
#parser.add_argument('--camNum', default=10, type=int, help='the number of came' )
opt = parser.parse_args()
print(opt )

#rs = opt.rs
#re = opt.re
mode = opt.mode
#camNum = opt.camNum
fileRoot = opt.fileRoot
#outputRoot = opt.outputRoot + '%d' % opt.camNum
outputRoot = opt.outputRoot + 'Real'
renderProgram = opt.renderProgram

shapes = glob.glob(osp.join(fileRoot, mode, 'Shape*') )
#for n in range(rs, min(re, len(shapes)) ):
for n, root in enumerate(shapes):
    if opt.shapeId is not None:
        if int(opt.shapeId) != n:
            continue
    #shapeRoot = osp.join(fileRoot, mode, 'Shape__%d' % n)
    #print('%d/%d: %s' % (n, min(re, len(shapes) ), shapeRoot ) )
    shapeRoot = root
    print('%d/%d: %s' % (n, len(shapes), shapeRoot ) )
    plyList = glob.glob(osp.join(root, '*.ply'))
    for plyPath in plyList:
        camNum = int(plyPath.split('_')[-1].split('.')[0])
        #camNum = int(shapeRoot.split('_')[-1])
        #shapeName = shapeRoot.split('__')[-1]

        shapeId = shapeRoot.split('/')[-1]
        outputDir = osp.join(outputRoot, mode, shapeId )

        if not osp.isdir(outputDir ):
            os.system('mkdir -p %s' % outputDir )
        else:
            print('Warning: output directory %s already exists' % (outputDir ) )
            results = glob.glob(osp.join(outputDir, 'imVH_%dtwoBounce_*.h5' % camNum ) )
            print(len(results ) )
            if len(results) == camNum:
                continue

        output = osp.join(outputDir, 'imVH_%d.rgbe' % camNum )

        xmlFile = osp.join(shapeRoot, 'imVH_%d.xml' % camNum )
        camFile = 'cam%d.txt' % camNum
        print(camNum)

        if opt.forceOutput:
            cmd1 = '%s -f %s -o %s -c %s -m 7 --forceOutput' % (renderProgram, xmlFile, output, camFile )
        else:
            cmd1 = '%s -f %s -o %s -c %s -m 7' % (renderProgram, xmlFile, output, camFile )

        os.system(cmd1 )

        results = glob.glob(osp.join(outputDir, 'imVH_%dtwoBounce_*.dat' % camNum ) )
        viewNum = len(results )
        cnt = 0
        for resName in results:
            cnt += 1
            print('Compressing %d/%d: %s' % (cnt, viewNum, resName ) )
            with open(resName, 'rb') as f:
                byte = f.read(4)
                height = struct.unpack('i', byte )[0]
                byte = f.read(4)
                width = struct.unpack('i', byte )[0]

                byte = f.read()
                twoBounce = np.array(struct.unpack(
                    str(width * height * 14) + 'f', byte ), dtype=np.float32 )
                twoBounce = twoBounce.reshape([height, width, 14])

                h5Name = resName.replace('.dat', '.h5')
                with h5py.File(h5Name, 'w') as fOut:
                    fOut.create_dataset('data', data = twoBounce, compression='lzf')
            os.system('rm %s' % resName )

            newh5Name = h5Name.replace('%dtwo' % camNum, 'two')
            os.system('mv %s %s' % (h5Name, newh5Name))


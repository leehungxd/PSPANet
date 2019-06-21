# -*- coding: utf-8 -*-
import sys, os
os.chdir('/home/lihang/projects/PSPNet/')

sys.path.insert(0, '/home/lihang/projects/PSPNet/python')
sys.path.insert(0, '/home/lihang/projects/codeTools')

import caffe
import cv2, sys, math
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
from codeTools import myTools
mytool = myTools.myTools()


netFile = '/home/lihang/projects/PSPNet/pspnet/models/pspnet50_ADE20K_473.prototxt'
caffeModel = '/home/lihang/projects/PSPNet/pspnet/jobs/cityscape/pspnet50/snapshots/pspnet50_cityscape_iter_4760.caffemodel'
testDir = '/home/lihang/data/cityscape/demoVideo/stuttgart_02'
# testDir = '/home/lihang/data/train_data/voc/VOCdevkit/VOC2012/JPEGImages'
testType = 'image'

if not os.path.exists(netFile):
    print('The net file does not exist!!!')
    exit()

if not os.path.exists(caffeModel):
    print('The caffe model does not exist!!!')
    exit()

net = caffe.Net(netFile, caffeModel, caffe.TEST)
caffe.set_device(0)
caffe.set_mode_gpu()

inputShape = net.blobs['data'].data.shape
outputShape = net.blobs['conv6_interp'].data.shape

def SegmentTest(image, labelMask_, thr):

    if image is None or labelMask_ is None:
        return False

    labelMask_ = cv2.cvtColor(labelMask_, cv2.COLOR_BGR2GRAY)
    inputSize = 256
    r,c,ch = image.shape

    imgInput = image.astype(np.float32)
    rows, cols, channels = imgInput.shape

    imgInput = cv2.resize(imgInput, (inputShape[3], inputShape[2]))
    imgInput = imgInput.transpose((2, 0, 1))
    imgInput = np.asarray([imgInput])

    out = net.forward_all(**{net.inputs[0]:imgInput})

    prediction = net.blobs['prob'].data[0].argmax(axis=0)
    predNp = np.array(prediction)

    prediction = np.squeeze(prediction).astype(np.uint8)
    prediction = 255 * prediction
    mask = cv2.resize(prediction, (image.shape[1], image.shape[0]))
    if mask.shape[0] == 1 or mask.shape[1] == 1:
        return False

    mask0 = np.zeros(mask.shape, dtype=np.uint8)
    mask1 = np.zeros(mask.shape, dtype=np.uint8)
    mask2 = np.zeros(mask.shape, dtype=np.uint8)
    predMask = np.zeros(mask.shape, dtype=np.uint8)
    labelMask = np.zeros(labelMask_.shape, dtype=np.uint8)
    mask0[mask < thr*255] = 0
    mask0[mask >= thr*255] = 0
    mask1[mask < thr*255] = 255
    mask1[mask >= thr*255] = 0
    mask2[mask < thr*255] = 0
    mask2[mask >= thr*255] = 0

    '''for dice accuracy compute'''
    predMask[mask < thr * 255] = 1
    predMask[mask >= thr * 255] = 0
    labelMask[labelMask_ < thr] = 1
    labelMask[labelMask_ >= thr] = 0

    accuarcy = mytool.SegDiceAccuracy(predMask, labelMask)

    maskMerged = cv2.merge([mask0, mask1, mask2])
    imageMask = cv2.addWeighted(image, 0.8, maskMerged, 0.2, 0)


    cv2.imshow('image', image)
    cv2.imshow('mask', imageMask)
    cv2.waitKey(1)
    return accuarcy

def SegmentTestVideo(image, imgFile):

    if image is None :
        return False

    imgInput = image.astype(np.float32)
    imgInput = imgInput[:, :, ::-1]
    mean = (104.008, 116.669, 122.675)
    imgInput -= mean
    rows, cols, channels = imgInput.shape

    imgInput = cv2.resize(imgInput, (inputShape[3], inputShape[2]))
    imgInput = imgInput.transpose((2, 0, 1))
    imgInput = np.asarray([imgInput])

    out = net.forward_all(**{net.inputs[0]:imgInput})

    prediction = net.blobs['conv6_interp'].data[0].argmax(axis=0)
    predNp = np.array(prediction)

    prediction = np.squeeze(prediction).astype(np.uint8)
    # prediction = 255 * prediction
    mask = cv2.resize(prediction, (image.shape[1], image.shape[0]))
    if mask.shape[0] == 1 or mask.shape[1] == 1:
        return False

    mask0 = np.zeros(mask.shape, dtype=np.uint8)
    mask1 = np.zeros(mask.shape, dtype=np.uint8)
    mask2 = np.zeros(mask.shape, dtype=np.uint8)

    # mask0[mask == 0] = 0
    # mask1[mask == 0] = 0
    # mask2[mask == 0] = 0
    # mask0[mask == 1] = 0
    # mask1[mask == 1] = 0
    # mask2[mask == 1] = 0
    # mask0[mask == 2] = 0
    # mask1[mask == 2] = 0
    # mask2[mask == 2] = 0
    # mask0[mask == 3] = 0
    # mask1[mask == 3] = 0
    # mask2[mask == 3] = 0
    # mask0[mask == 4] = 0
    # mask1[mask == 4] = 0
    # mask2[mask == 4] = 0
    # mask0[mask == 5] = 0
    # mask1[mask == 5] = 74
    # mask2[mask == 5] = 111
    # mask0[mask == 6] = 81
    # mask1[mask == 6] = 0
    # mask2[mask == 6] = 81
    mask0[mask == 0] = 128
    mask1[mask == 0] = 64
    mask2[mask == 0] = 128
    mask0[mask == 1] = 232
    mask1[mask == 1] = 35
    mask2[mask == 1] = 244
    # mask0[mask == 9] = 160
    # mask1[mask == 9] = 170
    # mask2[mask == 9] = 250
    # mask0[mask == 10] = 140
    # mask1[mask == 10] = 150
    # mask2[mask == 10] = 230
    mask0[mask == 2] = 70
    mask1[mask == 2] = 70
    mask2[mask == 2] = 70
    mask0[mask == 3] = 156
    mask1[mask == 3] = 102
    mask2[mask == 3] = 102
    mask0[mask == 4] = 153
    mask1[mask == 4] = 153
    mask2[mask == 4] = 190
    # mask0[mask == 14] = 180
    # mask1[mask == 14] = 165
    # mask2[mask == 14] = 180
    # mask0[mask == 15] = 100
    # mask1[mask == 15] = 100
    # mask2[mask == 15] = 150
    # mask0[mask == 16] = 90
    # mask1[mask == 16] = 120
    # mask2[mask == 16] = 150
    mask0[mask == 5] = 153
    mask1[mask == 5] = 153
    mask2[mask == 5] = 153
    # mask0[mask == 18] = 153
    # mask1[mask == 18] = 153
    # mask2[mask == 18] = 153
    mask0[mask == 6] = 30
    mask1[mask == 6] = 170
    mask2[mask == 6] = 250
    mask0[mask == 7] = 0
    mask1[mask == 7] = 220
    mask2[mask == 7] = 220
    mask0[mask == 8] = 35
    mask1[mask == 8] = 142
    mask2[mask == 8] = 107
    mask0[mask == 9] = 152
    mask1[mask == 9] = 251
    mask2[mask == 9] = 152
    mask0[mask == 10] = 180
    mask1[mask == 10] = 130
    mask2[mask == 10] = 70
    mask0[mask == 11] = 60
    mask1[mask == 11] = 20
    mask2[mask == 11] = 220
    mask0[mask == 12] = 0
    mask1[mask == 12] = 0
    mask2[mask == 12] = 255
    mask0[mask == 13] = 142
    mask1[mask == 13] = 0
    mask2[mask == 13] = 0
    mask0[mask == 14] = 70
    mask1[mask == 14] = 0
    mask2[mask == 14] = 0
    mask0[mask == 15] = 100
    mask1[mask == 15] = 60
    mask2[mask == 15] = 0
    # mask0[mask == 29] = 90
    # mask1[mask == 29] = 0
    # mask2[mask == 29] = 0
    # mask0[mask == 30] = 110
    # mask1[mask == 30] = 0
    # mask2[mask == 30] = 0
    mask0[mask == 16] = 100
    mask1[mask == 16] = 80
    mask2[mask == 16] = 0
    mask0[mask == 17] = 230
    mask1[mask == 17] = 0
    mask2[mask == 17] = 0
    mask0[mask == 18] = 32
    mask1[mask == 18] = 11
    mask2[mask == 18] = 119

    maskMerged = cv2.merge([mask0, mask1, mask2])

    if(0):
        maskFile = imgFile.replace('stuttgart_02', 'mask_02')
        cv2.imwrite(maskFile, maskMerged)
    else:
        # imageMask = cv2.addWeighted(image, 0.6, maskMerged, 0.4, 0)
        image_ = cv2.resize(image, (1000, 500))
        maskMerged_ = cv2.resize(maskMerged, (1000, 500))
        cv2.imshow('image', image_)
        cv2.imshow('mask', maskMerged_)
        cv2.waitKey(1)


if __name__ == '__main__':

    if testType == 'statisAP':
        imgFiles = mytool.GetFileList(testDir, 'jpg')
        accuacies = []
        for imgFile in tqdm(imgFiles):
            maskFile = '/home/lihang/data/train_data/bdd100k/seg/labels/val01/'+\
                       imgFile.split('/')[-1].split('.')[0]+'_train_id.png'

            image = cv2.imread(imgFile)
            labelMask = cv2.imread(maskFile)
            accu = SegmentTest(image, labelMask, 0.5)
            accuacies.append(accu)
        print(np.mean(accuacies))

    elif testType == 'video':
        testFile = '/caffe/test_data/car.avi'
        capture = cv2.VideoCapture(testFile)
        frameCount = 1
        if (capture.isOpened() == False):
            print('Error opening video stream or file!!!')

        while(capture.isOpened()):
            ret, frame = capture.read()
            if ret == True:
                if SegmentTestVideo(frame, 0.1) == False:
                    continue

    elif testType == 'image':
        imgFiles = mytool.GetFileList(testDir, 'png')

        for imgFile in tqdm(imgFiles):

            image = cv2.imread(imgFile)

            if SegmentTestVideo(image, imgFile) == False:
                continue









"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
from data import VOC_CLASSES as labelmap
import torch.utils.data as data
import matplotlib.pyplot as plt


from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/v2sfd_ohem4.pth',
                    type=str, help='Trained state_dict file path to open')
# parser.add_argument('--save_folder', default='eval/fddb_res', type=str,
#                     help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
# parser.add_argument('--cuda', default=True, type=str2bool,
#                     help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--im', default=None, type=str, help='Location of VOC root directory')

args = parser.parse_args()




class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff



def vis_detections(im, class_name, dets,neg, thresh=1):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:,-1]>-1)[0]
    # print(dets[:,-1])
    # print(inds)
    if len(inds) == 0:
        return

    # im = im[:, :, (2, 1, 0)]
    # im = im[:, :, (2, 0, 1)]
    # im = im.transpose(2, 0, 1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i,:4]
        score = dets[i,-1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=0.7)
            )

        ax.text(bbox[0]-5, bbox[1]-5,
               '{:.3f}'.format(score),
               bbox=dict(facecolor='blue', alpha=0.5),
               fontsize=5, color='white')
    inds = np.where(neg[:,-1]>-1)[0]
    # print(dets[:,-1])
    # print(inds)
    if len(inds) == 0:
        return

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = neg[i,:4]
        score = neg[i,-1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='green', linewidth=1.5)
            )

        # ax.text(bbox[0]-5, bbox[1]-5,
        #        '{:.3f}'.format(score),
        #        bbox=dict(facecolor='blue', alpha=0.5),
        #        fontsize=5, color='white')
    major_ticks = np.arange(0, 640, 16)
    ax.set_yticks(major_ticks)
    ax.set_xticks(major_ticks)
    ax.grid(True)
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    # plt.axis('off')
    plt.tight_layout()
    # plt.draw()
    plt.show()

    # plt.savefig('foo.png')



def test_net(net, im_path, mean, im_size=300, thresh=0.05):
    _t = {'im_detect': Timer(), 'misc': Timer()}
    tm = 0
    cnt = 0
    im = cv2.imread(im_path)
    h, w, channels = im.shape

    im = cv2.resize(im, (im_size, im_size)).astype(np.float32)
    im -= mean
    im = im.astype(np.float32)
    im = im[:, :, (2, 1, 0)]
    im = torch.from_numpy(im).permute(2, 0, 1)

    x = Variable(im.unsqueeze(0))

    _t['im_detect'].tic()
    #t0 = time.time()
    detections = net(x).data
    #t1 = time.time()
    #tm = t1 - t0 + tm
    # cnt += 1
    #print(tm)
    detect_time = _t['im_detect'].toc(average=False)
    print(detect_time)

    scale = torch.Tensor([w,h, w, h])
   # print(im_name,im.shape)
    for i in range(detections.size(1)):
        if i == 0:
            continue
        j = 0
        # print(detections.size())
        #fid.write(str(detections.size(1)) + '\n')
        cnt = 0
        scores = []
        pts = []
        #print(detections[0])
        while detections[0, i, j, 0] >= 0.5:
            cnt += 1

            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords =[pt[0], pt[1], pt[2], pt[3]]

            j += 1
            scores.append(score)
            pts.append(pt)
        pts = np.array(pts)
    print(scores,pts)
    pts = np.concatenate((pts, np.array([scores]).T), axis=1)

    neg_prior = np.loadtxt('gt.txt')
    neg = np.zeros(np.shape(neg_prior))
    neg[:,0] = neg_prior[:,0]
    neg[:,1] = neg_prior[:,1] 
    neg[:,2] = neg_prior[:,0] + neg_prior[:,2]
    neg[:,3] = neg_prior[:,1] + neg_prior[:,3]

    neg[:,0] = neg[:,0]/w*640.0 
    neg[:,2] = neg[:,2]/w*640.0 
    neg[:,1] = neg[:,1]/h*640.0 
    neg[:,3] = neg[:,3]/h*640.0 

    pts[:,0] = pts[:,0]/w*640.0 
    pts[:,2] = pts[:,2]/w*640.0 
    pts[:,1] = pts[:,1]/h*640.0 
    pts[:,3] = pts[:,3]/h*640.0 



    img = cv2.imread(im_path)
    img = cv2.resize(img, (640, 640))
    img = img[:, :, (2, 1, 0)]
    class_name = 'face'
    vis_detections(img, class_name, pts,neg, thresh=0.8)




if __name__ == '__main__':
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background

    net = build_ssd('test', 640, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model,map_location=lambda storage, loc: storage))
    # torch.load(args.trained_model)
    net.eval()
    print('Finished loading model!')
    # if args.cuda:
    #     net = net.cuda()
    #     cudnn.benchmark = True
    mean = np.array((105.0010, 111.6685, 121.4912), dtype=np.float32)
    # evaluation
    test_net(net, args.im, mean, 640, thresh=args.confidence_threshold)


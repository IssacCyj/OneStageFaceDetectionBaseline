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
parser.add_argument('--trained_model', default='weights/v2sfd_ohem2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/fddb_res', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')



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

def get_im_fddb(data_dir):
    imdb = []
    nfold = 10
    for n in range(nfold):
      file_name = 'FDDB-folds/FDDB-fold-%02d.txt' % (n + 1)
      file_name = os.path.join(data_dir, file_name)
      fid = open(file_name, 'r')
      image_names = []
      for im_name in fid:
        image_names.append(im_name.strip('\n'))
      imdb.append(image_names)

    return imdb

def test_net(net, cuda, mean, im_size=300, thresh=0.05):
    _t = {'im_detect': Timer(), 'misc': Timer()}
    tm = 0
    cnt = 0
    root = '/home/yujia.chen/face-adversarial-frcnn/data/FDDB'
    imdb = get_im_fddb(root)
    nfold = len(imdb)
    for i in range(nfold):
        image_names = imdb[i]
        print('processing file %d\n'%i)
        # detection file
        dets_file_name = os.path.join(args.save_folder, 'ssdBased_fold-%02d-out.txt' % (i + 1))
        fid = open(dets_file_name, 'w')
        #sys.stdout.write('%s ' % (i + 1))

        for idx, im_name in enumerate(image_names):
            im = cv2.imread(os.path.join(root, 'originalPics', im_name + '.jpg'))
            h, w, channels = im.shape

            im = cv2.resize(im, (im_size, im_size)).astype(np.float32)
            im -= mean
            im = im.astype(np.float32)
            im = im[:, :, (2, 1, 0)]
            im = torch.from_numpy(im).permute(2, 0, 1)

            x = Variable(im.unsqueeze(0))
            if args.cuda:
               x = x.cuda()
            _t['im_detect'].tic()
            #t0 = time.time()
            detections = net(x).data
            #t1 = time.time()
            #tm = t1 - t0 + tm
            cnt += 1
            #print(tm)
            detect_time = _t['im_detect'].toc(average=False)
            #print(detect_time)
            fid.write(im_name + '\n')

            scale = torch.Tensor([w,h, w, h])
           # print(im_name,im.shape)
            for i in range(detections.size(1)):
                if i == 0:
                    continue
                j = 0
                #print(detections.size())
                #fid.write(str(detections.size(1)) + '\n')
                cnt = 0
                scores = []
                pts = []
                #print(detections[0])
                while detections[0, i, j, 0] >= 0.2:
                    cnt += 1
                    # if pred_num == 0:
                    #     with open(filename, mode='a') as f:
                    #         f.write('PREDICTIONS: '+'\n')
                    score = detections[0, i, j, 0]
                    # label_name = labelmap[i-1]
                    pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                    coords =[pt[0], pt[1], pt[2], pt[3]]
                    # with open(filename, mode='a') as f:
                    #     f.write(str(pred_num)+' score: ' +
                    #             str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                    j += 1
                    scores.append(score)
                    pts.append(pt)
                pts = np.array(pts)
                #print(pts,type(pts))
                fid.write(str(cnt) + '\n')
                for n in range(len(scores)):
                    fid.write('%f %f %f %f %f\n' % (pts[n,0], pts[n,1], pts[n,2] - pts[n,0], pts[n,3]-pts[n,1], scores[n]))
    print('Average time per image: %.4f'%(tm/cnt))






if __name__ == '__main__':
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 640, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    mean = np.array((105.0010, 111.6685, 121.4912), dtype=np.float32)
    # evaluation
    test_net(net, args.cuda, mean, 640, thresh=args.confidence_threshold)


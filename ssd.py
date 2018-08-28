import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2
import os
import numpy as np

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        #self.size = 640

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm3 = L2Norm(256, 10)
        self.L2Norm4 = L2Norm(512, 8)
        self.L2Norm5 = L2Norm(512, 5)
        self.cnn4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.cnn1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.cnn512_256 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.cnn512 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.cnn512_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.cnn256 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.cnn1024_512 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.25)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        pre_sources = list()
        loc = list()
        conf = list()

#************
        # # apply vgg up to conv3_3 relu
        # for k in range(16):
        #     x = self.vgg[k](x)

        # s = self.L2Norm3(x)
        # s = F.relu(self.cnn4(s)) #256->128
        # s3 = F.relu(self.cnn1(s)) #128 ->128
        # s57 = F.relu(self.cnn2(s)) #128 -> 64
        # s5 = F.relu(self.cnn3(s57)) #64 64
        # s71 = F.relu(self.cnn3(s57))# 64 64
        # s72 = F.relu(self.cnn3(s71)) # 64 64
        # c = torch.cat((s3,s5,s72), 1)
        # sources.append(c)
#************

# #************
        # apply vgg up to conv3_3 relu
        for k in range(16):
            x = self.vgg[k](x)

        s = self.L2Norm3(x)
        pre_sources.append(s)
# #************

        # apply vgg up to conv4_3 relu
        for k in range(16,23):
            x = self.vgg[k](x)

        s = self.L2Norm4(x)
        # s = self.L2Norm(x)
        pre_sources.append(s)

        # apply vgg up to conv5_3 relu
        for k in range(23,30):
            x = self.vgg[k](x)

        s = self.L2Norm5(x)
        # s = self.L2Norm(x)
        pre_sources.append(s)


        # assert(len(self.vgg) == 31)
        # apply vgg up to pool5
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        pre_sources.append(x)




        for i in range(3):
            if i == 0:
                s = self.cnn512_256(pre_sources[i+1])
                s = self.upsample(s)
                src = s + pre_sources[i]
                src = self.cnn256(src)
            elif i == 1:
                s = self.cnn512(pre_sources[i+1])
                s = self.upsample(s)
                src = s + pre_sources[i]
                src = self.cnn512_2(src)
            else:
                s = self.cnn1024_512(pre_sources[i+1])
                s = self.upsample(s)
                src = s + pre_sources[i]
                src = self.cnn512_2(src)
            sources.append(src)

        sources.append(x)



        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
               # print(x.size())
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
       
      
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #print(loc.size())
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    #print(layers)
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    # pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # layers += [pool5, conv6,
               # nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    #vgg_source = [24, -2]
    vgg_source = [14, 21, 28, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 4):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '640': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#    '640': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 256],
    '640': [256, 'S', 512, 128, 'S', 256],

}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '640': [3,3,3,3,3,3],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    #if size != 300:
    #    print("Error: Sorry only SSD300 is supported currently!")
    #    return

    return SSD(phase, *multibox(vgg(base[str(size)], 3),
                                add_extras(extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)

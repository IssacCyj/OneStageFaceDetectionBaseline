"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

#VOC_CLASSES = (  # always index 0
#    'aeroplane', 'bicycle', 'bird', 'boat',
#    'bottle', 'bus', 'car', 'cat', 'chair',
#    'cow', 'diningtable', 'dog', 'horse',
#    'motorbike', 'person', 'pottedplant',
#    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASSES = ( 'face', )

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, transform=None, target_transform=None,
                 dataset_name='WIDER'):
        self.root = root   
        self.image_set = image_sets   #[('WIDER','train')]
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        # self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        # self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        # self.ids = list()
        # for (year, name) in image_sets:
        #     rootpath = os.path.join(self.root, 'VOC' + year)
        #     for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))
        self.all_boxes, self.all_names = self._load_image_set_index()


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        #print(im.shape)
        return im, gt

    def __len__(self):
        return len(self.all_names)



    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = 'wider_face_train_annot.txt'
        image_set_file = os.path.join(self.root, image_set_file)
        #print('annopath:%s'%image_set_file)
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        image_index = []
        all_boxes = []
        
        with open(image_set_file) as f:
            # print len(f.lines())
            lines = f.readlines()

            idx = 0
            while idx < len(lines):
                image_name = lines[idx].split('\n')[0]
                image_name = os.path.join('WIDER_train/images', image_name)
                # print image_name
                image_ext = os.path.splitext(image_name)[1].lower()
                # print image_ext
                assert(image_ext == '.png' or image_ext == '.jpg' or image_ext == '.jpeg')

                image = Image.open(os.path.join(self.root, image_name))
                imw = image.size[0]
                imh = image.size[1]

                idx += 1
                num_boxes = int(lines[idx])
                # print num_boxes

                boxes = np.zeros((num_boxes, 5), dtype=np.float64)

                for i in range(num_boxes):
                    idx += 1
                    coor = list(map(float, lines[idx].split()))

                    x1 = min(max(coor[0], 0), imw - 1)
                    y1 = min(max(coor[1], 0), imh - 1)
                    x2 = min(max(x1 + coor[2] - 1, 0), imw - 1)
                    y2 = min(max(y1 + coor[3] - 1, 0), imh - 1)

                    if np.isnan(x1):
                        x1 = -1

                    if np.isnan(y1):
                        y1 = -1

                    if np.isnan(x2):
                        x2 = -1

                    if np.isnan(y2):
                        y2 = -1
                        
                    # label_idx = self.class_to_ind['face']
                    label_idx = 0
                    boxes[i, :] = [x1, y1, x2, y2, label_idx]


                widths = boxes[:, 2] - boxes[:, 0] + 1
                heights = boxes[:, 3] - boxes[:, 1] + 1
                keep_idx = np.where(np.bitwise_and(widths > 5, heights > 5))

                #print('keepidx:%s'%keep_idx)
                #if all bboxes(width or height) are smaller than 5, then omit this image 
                if len(keep_idx[0]) <= 0:
                    idx += 1
                    continue

                boxes = boxes[keep_idx]

               # print('boxes:%s'%boxes)
               # print('imw:%d'%imw)

                boxes[:,0] /= imw
                boxes[:,1] /= imh
                boxes[:,2] /= imw
                boxes[:,3] /= imh



                if not (boxes[:, 2] >= boxes[:, 0]).all():
                    print (boxes)
                    print (image_name)

                #print (boxes.tolist())
                assert (boxes[:, 2] >= boxes[:, 0]).all()
                assert (boxes[:, 3] >= boxes[:, 1]).all()
                all_boxes.append(boxes.tolist())
                image_index.append(image_name)

                idx += 1        

            assert(idx == len(lines))
        #print('shapes:%s'%np.array(all_boxes).shape)
        #print(all_boxes)
        return all_boxes, image_index


    def pull_item(self, index):
        img_name = self.all_names[index]

        # target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(os.path.join(self.root,img_name))
        height, width, channels = img.shape

        #print('imgpath:%s'%os.path.join(self.root,img_name))
        #print('index: ',index)
        if self.target_transform is not None:
            # target = self.target_transform(target, width, height)
            target_ = self.all_boxes[index]
            #target = np.array(target_)
            #print('target:%s'%target_)
        if self.transform is not None:
            target = np.array(target_)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
         
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


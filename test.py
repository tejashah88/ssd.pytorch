from __future__ import print_function
import sys
import os
import argparse
import random

from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_ROOT, VOC_CLASSES, CUSTOM_CLASSES, MEANS
from ssd import build_ssd

from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--use_custom', default=False, type=str2bool,
                    help='If specified, use the custom VOC Detection implementation')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.use_custom:
    from data import CUSTOM_CLASSES as labelmap
else:
    from data import VOC_CLASSES as labelmap


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1

def test_random_img(save_folder, net, cuda, testset, transform, thresh):
    import matplotlib.pyplot as plt

    i = random.randint(0, len(testset))
    img = testset.pull_image(i)
    img_id, annotation = testset.pull_anno(i)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    img_copy = img.copy()

    print('\nGROUND TRUTH FOR: '+img_id+'\n')
    for box in annotation:
        print('label: '+' || '.join(str(b) for b in box)+'\n')

    cv2.rectangle(img_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)

    if cuda:
        x = x.cuda()

    y = net(x)      # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([img.shape[1], img.shape[0],
                         img.shape[1], img.shape[0]])
    pred_num = 0

    for i in range(detections[:1].size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            if pred_num == 0:
                print('PREDICTIONS: '+'\n')
            score = detections[0, i, j, 0]
            label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])

            cv2.rectangle(img_copy, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 3)
            pred_num += 1

            print(str(pred_num)+' label: '+label_name+' score: ' +
                    str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
            j += 1

    cv2.imwrite('output.jpg', img_copy)

def test_voc():
    # load net
    num_classes = len(CUSTOM_CLASSES if args.use_custom else VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')

    # load data
    if args.use_custom:
        custom_class_to_ind = dict(zip(CUSTOM_CLASSES, range(len(CUSTOM_CLASSES))))
        testset = VOCDetection(
            root=args.voc_root,
            image_sets=[('2019', 'test')],
            dataset_name='VOC2019',
            transform=None,
            target_transform=VOCAnnotationTransform(class_to_ind=custom_class_to_ind))
    else:
        testset = VOCDetection(
            root=args.voc_root,
            image_sets=[('2007', 'test')],
            dataset_name='VOC0712',
            transform=None,
            target_transform=VOCAnnotationTransform())

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    test_random_img(args.save_folder, net, args.cuda, testset,
                    BaseTransform(net.size, MEANS),
                    thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()

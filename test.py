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

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained-model', dest='trained_model', default='weights/ssd_300_VOC0712.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual-threshold', dest='visual_threshold', default=0.6, type=float, help='Final confidence threshold')
parser.add_argument('--voc-root', dest='voc_root', default=VOC_ROOT, help='Location of VOC root directory')

parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use CUDA to train model (default)')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Do not use CUDA to train model')
parser.set_defaults(cuda=True)

parser.add_argument('--custom-voc', dest='use_custom', action='store_true', help='Use a custom VOC-like dataset')
parser.add_argument('--standard-voc', dest='use_custom', action='store_false', help='Use the standard VOC dataset (default)')
parser.set_defaults(use_custom=False)

args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


if args.use_custom:
    from data import CUSTOM_CLASSES as labelmap
else:
    from data import VOC_CLASSES as labelmap


def test_random_img(net, cuda, testset, transform, thresh):
    i = random.randint(0, len(testset))
    img = testset.pull_image(i)
    height, width = img.shape[:2]
    img_id, annotation = testset.pull_anno(i)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)

    img_copy = img.copy()

    print(f'GROUND TRUTH FOR: {img_id}')
    for box in annotation:
        print('label: '+' || '.join(str(b) for b in box[:4])+'\n')

    cv2.rectangle(img_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)

    if cuda:
        x = x.cuda()

    x = Variable(x.unsqueeze(0))
    y = net(x)      # forward pass
    detections = y.data

    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])

    i = 1
    j = 0
    score = detections[0, i, j, 0]
    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
    coords = (pt[0], pt[1], pt[2], pt[3])

    cv2.rectangle(img_copy, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 3)
    print('score: ' + str(score) + ' ' + ' || '.join(str(c) for c in coords))

    cv2.imshow('output', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_voc():
    # load net
    num_classes = len(CUSTOM_CLASSES if args.use_custom else VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD

    if args.cuda:
        net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cuda')))
    else:
        net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu')))

    net.eval()
    print('Finished loading model!')

    # load data
    if args.use_custom:
        custom_class_to_ind = dict(zip(CUSTOM_CLASSES, range(len(CUSTOM_CLASSES))))
        testset = VOCDetection(
            root=args.voc_root,
            image_sets=[('2019', 'test')],
            dataset_name='VOC2019',
            transform=BaseTransform(300, MEANS),
            target_transform=VOCAnnotationTransform(class_to_ind=custom_class_to_ind))
    else:
        testset = VOCDetection(
            root=args.voc_root,
            image_sets=[('2007', 'test')],
            dataset_name='VOC0712',
            transform=BaseTransform(300, MEANS),
            target_transform=VOCAnnotationTransform())

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    test_random_img(net, args.cuda, testset,
                    BaseTransform(300, MEANS),
                    thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()

from __future__ import print_function
import sys
import os
import argparse
import random

from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_ROOT, CUSTOM_CLASSES as labelmap, MEANS
from ssd import build_ssd

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data

num_classes = len(labelmap) + 1 # +1 background
custom_class_to_ind = dict(zip(labelmap, range(len(labelmap))))

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def predict(input_img, transform, net, cuda, debug_predictions='none'):
    height, width = input_img.shape[:2]
    x = torch.from_numpy(transform(input_img)[0]).permute(2, 0, 1)

    if cuda:
        x = x.cuda()

    x = Variable(x.unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data

    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])

    i = 1
    j = 0
    best_pt = (detections[0, i, j, 1:] * scale).cpu().numpy()

    if debug_predictions == 'full':
        print('PREDICTIONS:')
        for i in range(detections.size(1)):
            while detections[0, i, j, 0] >= 0.05:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                print('  BOX: ' + str(np.rint(pt)) + ' => \t' + '%.4f' % detections[0, i, j, 0].item())
                cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, int(255 * detections[0, i, j, 0]), 0), 2)
                j += 1
    else:
        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
        if debug_predictions == 'simple':
            print('Predicted box: \t' + str(np.rint(pt).astype(int)))
            print('Confidence: \t%.4f' % detections[0, i, j, 0].item())
        cv2.rectangle(input_img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 255, 0), 2)
    return input_img, best_pt

def run_inference(model_file, cuda=True):
    net = build_ssd('test', 300, num_classes) # initialize SSD

    if cuda:
        net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cuda')))
    else:
        net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu')))

    net = net.eval()

    if cuda:
        net = net.cuda()
        cudnn.benchmark = True

    print('Finished loading model!')

    def predict_simple(input_img):
        output, pt = predict(input_img, BaseTransform(300, MEANS), net, cuda)
        return output, pt

    return predict_simple

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Inference With Pytorch')
    parser.add_argument('--input', type=str, help='Input image to run inference on')
    parser.add_argument('--trained-model', dest='trained_model', default='weights/ssd_300_VOC0712.pth', type=str, help='Trained state_dict file path to open')

    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use CUDA to train model (default)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Do not use CUDA to train model')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()


    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


    predict_simple = run_inference(args.trained_model, args.cuda)
    input_img = cv2.imread(args.input)
    output_img, pt = predict_simple(input_img)


    cv2.imshow('output', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
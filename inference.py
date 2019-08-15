from __future__ import print_function
import sys
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_ROOT, CUSTOM_CLASSES as labelmap
import torch.utils.data as data
from ssd import build_ssd
import cv2
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

num_classes = len(labelmap) + 1 # +1 background
custom_class_to_ind = dict(zip(labelmap, range(len(labelmap))))

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def predict(frame, transform, net):
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1).cuda()
    x = Variable(x.unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data

    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])

    i = 1
    j = 0
    # for i in range(detections.size(1)):
    # while detections[0, i, j, 0] >= 0.95:
    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
    cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), COLORS[i % 3], 2)
        # j += 1
    return frame, pt

def run_inference(model_file):
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(model_file))
    net = net.eval()
    net = net.cuda()
    cudnn.benchmark = True
    print('Finished loading model!')

    def predict_simple(input_file):
        input_img = cv2.imread(input_file)
        output, pt = predict(input_img, BaseTransform(net.size, MEANS), net)
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

    predict_simple = run_inference(args.trained_model)
    output_img, pt = predict_simple(args.input)
    cv2.imwrite('output-inference.jpg', output_img)
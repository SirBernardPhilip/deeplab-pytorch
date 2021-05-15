#!/usr/bin/env python
# coding: utf-8
#
# Authors: Bernat Felip, Kazuto Nakashima
# URL:    https://sirbernardphilip.github.io, https://kazuto1011.github.io
# Date:   14 May 2021

from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from libs.models import *
from libs.utils import DenseCRF


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
    return classes


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap


@click.group()
@click.pass_context
def main(ctx):
    """
    Demo with a trained model
    """

    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-ip",
    "--in-path",
    type=click.Path(exists=True),
    required=True,
    help="Images to be processed",
)
@click.option(
    "-mp",
    "--map-path",
    type=click.Path(exists=True),
    required=True,
    help="Images to be processed",
)
@click.option(
    "-o",
    "--out-path",
    type=click.Path(exists=True),
    required=True,
    help="Output path of JSONs",
)
def multiple(config_path, in_path, map_path, out_path):
    """
    Inference from multiple images
    """

    # Setup
    CONFIG = OmegaConf.load(config_path)

    classes = get_classtable(CONFIG)
    count = 0;
    # Inference
    tours = sorted(os.listdir(in_path))
    for tour in tqdm(tours, leave=False):
        images_path = os.path.join(in_path, tour)
        images = sorted(os.listdir(images_path))
        for image_path in tqdm(images, leave=False):
            count = count + 1
            path = os.path.join(images_path, image_path)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (1024,512))
            #print(imageAux.shape)20484096
            imageAux = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (513,256))
            labelmap = np.load(os.path.join(map_path, tour, image_path[:-5]+'_map.npy'))
            labels = np.unique(labelmap)

            out_image_path = os.path.join(out_path, tour)
            if(not os.path.exists(out_image_path)):
                os.mkdir(out_image_path)
            # Show result for each class
            for i, label in enumerate(labels):
                mask = np.zeros([256,513,3], np.uint8)
                mask[:,:,0] = ((labelmap == label)*255).astype(np.uint8)
                mask[:,:,1] = ((labelmap == label)*255).astype(np.uint8)
                mask[:,:,2] = ((labelmap == label)*255).astype(np.uint8)
                #cv2.imshow("First", mask)
                cv2.waitKey(0)
                mask = cv2.resize(mask, (1024,512))
                #cv2.imshow("Second", mask)
                cv2.waitKey(0)
                gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                #cv2.imshow("Third", gray_mask)
                cv2.waitKey(0)
                ret, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
                thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for c in contours:
                     x,y,w,h = cv2.boundingRect(c)
                     if w*h > 400:
                         cv2.putText(image, classes[label],(x+20,y+10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1,cv2.LINE_AA)
                         cv2.drawContours(image, [c], 0, (0, 255, 0), 1)
                #cv2.drawContours(image, contours, 0, (0, 255, 0), 10)
                #M = cv2.moments(contours[0])
                #print("center X : '{}'".format(round(M['m10'] / M['m00'])))
                #print("center Y : '{}'".format(round(M['m01'] / M['m00'])))
                #cv2.circle(image, (round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])), 5, (0, 255, 0), -1)
                #cv2.imshow(classes[label],image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            print(os.path.join(out_image_path, image_path[:-5]+'.jpeg'))
            cv2.imwrite(os.path.join(out_image_path, image_path[:-5]+'.jpeg'), image)

            #for i, label in enumerate(labels):
                #labelmap = cv2.resize(labelmap, (4096,2048))
                #print(labelmap)
              #  mask = labelmap == label
               # print(mask)
                #mask = cv2.resize(mask, (409,2048))
                
            #cv2.imshow('image',image)
            #cv2.waitKey(0)

                #plt.imshow(mask.astype(np.float32), alpha=0.5)
           # plt.axis('off')
            #plt.show()

if __name__ == "__main__":
    main()

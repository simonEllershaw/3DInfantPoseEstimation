from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import random
import cv2
from torchvision import transforms
import os
import sys
import copy
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import DataSets.transforms
from DataSets.TargetType import TargetType
import DataSets.config as cfg


class JointsDataset(Dataset):
    def __init__(self, mode, numJoints):
        # Preprocessing constants
        self.mode = mode
        self.imageSize = cfg.generic["imageSize"]
        self.numJoints = numJoints
        self.db = []

    def __len__(self):
        return len(self.db)

    def _get_db(self):
        pass

    def __getitem__(self, idx):
        pass

    def visualiseSample(self, sample):
        pass

    def getBboxCentreAndScaleFrom2DJointPos(self, jointPos2D, imageSize):
        # Bounding box corners given by padding from max and min 2D joint
        # coordinates in each axis
        padding = 100
        # Switch image size x and y
        imageSize = np.array(imageSize)
        imageSize[0], imageSize[1] = imageSize[1], imageSize[0]

        jointsNoFillValues = jointPos2D[(jointPos2D[:, 0] != -1) & (jointPos2D[:, 1] != -1)]
        x1 = np.min(jointsNoFillValues[:, 0]) - padding
        y1 = np.min(jointsNoFillValues[:, 1]) - padding
        x2 = np.max(jointsNoFillValues[:, 0]) + padding
        y2 = np.max(jointsNoFillValues[:, 1]) + padding

        # CHECK
        # Bbox is contrained to image dimensions
        bbox = np.array([x1, y1, x2, y2])
        bbox = np.clip(bbox, 0, np.concatenate((imageSize, imageSize)))

        # centre = (bbox[2:] + bbox[:2]) / 2
        bboxDimensions = bbox[2:] - bbox[:2]
        scale = max(bboxDimensions) / 200  # scale in relation to 200px
        centre = (bbox[2:] + bbox[:2]) / 2

        return centre, scale

    def preProcessImage(self, image, center, scale):
        # Get rotation angle (random if training 0 otherwise)
        orgImageWidth = -1
        if self.mode == "train":
            (
                image,
                center,
                rotation,
                orgImageWidth,
            ) = JointsDataset.randomRotationAndFlip(image, center, orgImageWidth)
        else:
            rotation = 0

        trans = DataSets.transforms.get_affine_transform(
            center, scale, rotation, self.imageSize
        )
        image = cv2.warpAffine(
            image, trans, (self.imageSize, self.imageSize), flags=cv2.INTER_LINEAR
        )
        image = transforms.ToTensor()(image)

        return image, trans, orgImageWidth

    def randomRotationAndFlip(image, center, orgImageWidth, rotation_factor=45):
        rf = rotation_factor
        rotation = (
            np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
            if random.random() <= 0.5
            else 0
        )
        if random.random() <= 0.5:
            image = image[:, ::-1, :]
            orgImageWidth = np.shape(image)[1]
            center[0] = np.shape(image)[1] - center[0] - 1
        return image, center, rotation, orgImageWidth

    def preProcessPixelLocations(anno, trans, orgImageWidth):
        # If image has been flipped orgImageWidth is supplied so joints cant be flipped also
        if orgImageWidth > -1:
            anno[:, 0] = orgImageWidth - anno[:, 0]

        # Transform joint position inaccordance to image transform
        for i in range(len(anno)):
            anno[i, 0:2] = DataSets.transforms.affine_transform(anno[i, 0:2], trans)
        return anno

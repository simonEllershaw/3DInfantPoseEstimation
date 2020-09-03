from torch.utils.data import Dataset
import numpy as np
import cv2
from torchvision import transforms
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

import DataSets.Utils.Transforms
import DataSets.Utils.Config as cfg


class JointsDataset(Dataset):
    """
        1st Level of dataset hirarchy implemented in this project
        Defines interface and contains generic pre-processing methods
        that are used by multiple child classes
    """

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

    def jointsToMPIIFormat(self, jointArray):
        MPIIFormat = []
        for i in range(len(self.MPIIMapping)):
            MPIIFormat.append(jointArray[self.MPIIMapping[i]])
        return np.array(MPIIFormat)

    def getBboxCentreAndScaleFrom2DJointPos(self, jointPos2D, imageSize):
        # Switch image size x and y
        imageSize = np.array(imageSize)
        imageSize[0], imageSize[1] = imageSize[1], imageSize[0]

        # Fill values do not bound subject so are removed
        jointsNoFillValues = jointPos2D[
            (jointPos2D[:, 0] != -1) & (jointPos2D[:, 1] != -1)
        ]

        # Min and max joint coordinates in each dimension
        # Padding added as terminal keypoints not always
        # at end of subject e.g. wrist keypoint
        padding = 100
        x1 = np.min(jointsNoFillValues[:, 0]) - padding
        y1 = np.min(jointsNoFillValues[:, 1]) - padding
        x2 = np.max(jointsNoFillValues[:, 0]) + padding
        y2 = np.max(jointsNoFillValues[:, 1]) + padding

        # Bbox is contrained to image dimensions
        bbox = np.array([x1, y1, x2, y2])
        bbox = np.clip(bbox, 0, np.concatenate((imageSize, imageSize)))

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

        # Transform image
        trans = DataSets.Utils.Transforms.get_affine_transform(
            center, scale, rotation, self.imageSize
        )
        image = cv2.warpAffine(
            image, trans, (self.imageSize, self.imageSize), flags=cv2.INTER_LINEAR
        )
        image = transforms.ToTensor()(image)

        return image, trans, orgImageWidth

    def randomRotationAndFlip(image, center, orgImageWidth, rotation_factor=45):
        # rf = rotation_factor
        rotation = 0  # (
        #     np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        #     if random.random() <= 0.5
        #     else 0
        # )
        # if random.random() <= -1:
        #     image = image[:, ::-1, :]
        #     orgImageWidth = np.shape(image)[1]
        #     center[0] = np.shape(image)[1] - center[0] - 1
        return image, center, rotation, orgImageWidth

    def preProcessPixelLocations(anno, trans, orgImageWidth):
        # If image has been flipped orgImageWidth is supplied so joints cant be flipped also
        if orgImageWidth > -1:
            anno[:, 0] = orgImageWidth - anno[:, 0]

        # Transform joint position inaccordance to image transform
        for i in range(len(anno)):
            anno[i, 0:2] = DataSets.Utils.Transforms.affine_transform(anno[i, 0:2], trans)
        return anno

import torch
import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.patches as patches

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.JointsDataset import JointsDataset


class BboxDataset(JointsDataset):
    def __init__(self, mode, numJoints):
        super().__init__(mode, numJoints)

    def _get_db(self):
        pass

    def generateBboxSample(bboxTarget, imagePath):
        return {"target": bboxTarget, "imagePath": imagePath}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        anno = copy.deepcopy(self.db[idx])

        image = np.array(Image.open(anno["imagePath"]))
        orgImageShape = np.shape(image)

        # Centre and scale are defined by image shape
        centre = np.array(image.shape[:-1]) // 2
        scale = np.max(image.shape[:-1]) / 200

        meta = {
            "imagePath": anno["imagePath"],
            "centre": centre,
            "scale": scale,
        }

        target = anno["target"]
        centre = np.array(image.shape[:-1]) // 2
        scale = np.max(image.shape[:-1]) / 200
        # Rescales to imageSize x imageSize with random rotation
        image, trans, orgImageWidth = self.preProcessImage(
            image, [centre[1], centre[0]], scale
        )

        # Transform bbox ground truth according
        corners = self.getCornersFromBoundingBox(np.array(target["boxes"]))
        transformedCorners = self.preProcessPixelLocations(
            corners, trans, orgImageWidth
        )
        target["boxes"] = self.getBoundingBoxFromCorners(transformedCorners)

        return image, target, meta

    def getCornersFromBoundingBox(self, bbox):
        corners = np.reshape(bbox, (-1, 2))
        topRight = np.array([corners[1][0], corners[0][1]])
        bottomLeft = np.array([corners[0][0], corners[1][1]])
        return np.vstack((corners, topRight, bottomLeft))

    def getBoundingBoxFromCorners(self, corners):
        topLeft = np.amin(corners, axis=0)
        bottomRight = np.amax(corners, axis=0)
        return torch.tensor(
            [topLeft[0], topLeft[1], bottomRight[0], bottomRight[1]],
            dtype=torch.float32,
        ).view(1, 4)

    def getGroundTruthBoundingBox(self, maskPath, paddingPixels):
        mask = np.array(Image.open(maskPath))

        y1, y2 = self.getMaxAndMinMaskPixelAlongAxis(mask, paddingPixels, axis=1)
        x1, x2 = self.getMaxAndMinMaskPixelAlongAxis(mask, paddingPixels, axis=0)

        return np.array([x1, y1, x2, y2])

    def getMaxAndMinMaskPixelAlongAxis(self, mask, paddingPixels, axis):
        flattenedSum = np.sum(mask, axis=axis)
        maskedSum = flattenedSum != 0
        minPixel = maskedSum.argmax() - paddingPixels
        maxPixel = (
            maskedSum.shape[0] - np.flip(maskedSum, axis=0).argmax() - 1 + paddingPixels
        )
        return minPixel, maxPixel

    def collate(batch):
        return tuple(zip(*batch))

    def visualiseSample(self, sample, ax, targetType):
        image, target, meta = sample
        plt.tight_layout()
        ax.set_title(meta["imagePath"])

        bbox = target["boxes"][0]
        x = bbox[0]
        y = bbox[1]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        rect = patches.Rectangle((x, y), width, height, edgecolor="b", fill=False)
        ax.add_patch(rect)

        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()
        ax.imshow(image)

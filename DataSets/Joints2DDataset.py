from abc import ABC, abstractmethod
import torch
from PIL import Image
import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.JointsDataset import JointsDataset
import DataSets.visualization as vis

class Joints2DDataset(JointsDataset):
    def __init__(self, mode, numJoints):
        # Preprocessing constants
        JointsDataset.__init__(self, mode, numJoints)
        self.rotation_factor = 45
        self.heatmapSize = 64
        self.sigma = 2

    def _get_db(self):
        pass

    def generateSample(joints2D, imagePath, scale, centre, PCKhThreshold):
        return {
            "joints2D": joints2D,
            "imagePath": imagePath,
            "scale": scale,
            "centre": centre,
            "2DPCKhThreshold": PCKhThreshold
        }

    @abstractmethod
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        anno = copy.deepcopy(self.db[idx])

        # Get data from anno
        image = np.array(Image.open(anno["imagePath"]))
        orgImageShape = np.shape(image)

        if anno["centre"] is None or anno["scale"] is None:
            anno["centre"], anno["scale"] = JointsDataset.getBboxCentreAndScaleFrom2DJointPos(
                anno["joints2D"], orgImageShape[:-1]
            )

        meta = {
            "imagePath": anno["imagePath"],
            "joints2D": anno["joints2D"].copy(),
            "centre": anno["centre"],
            "scale": anno["scale"],
            "2DPCKhThreshold": anno["2DPCKhThreshold"]
        }

        # Preprocessing
        meta["visJoints"] = self.calcVisJoints(anno["joints2D"], orgImageShape)
        image, trans, orgImageWidth = self.preProcessImage(
            image, anno["centre"], anno["scale"]
        )
        processedJoints2D = JointsDataset.preProcessPixelLocations(
            anno["joints2D"], trans, orgImageWidth
        )
        target, meta["visJoints"] = self.generateJointHeatmaps(
            processedJoints2D, self.imageSize, self.heatmapSize, meta["visJoints"]
        )

        return image, target, meta

    def generateJointHeatmaps(self, joints, imageSize, heatmapSize, visJoints):
        numJoints = np.shape(joints)[0]

        target_weight = np.ones(numJoints, dtype=np.float32)
        if visJoints is not None:
            target_weight[:] = visJoints[:]

        target = np.zeros((numJoints, heatmapSize, heatmapSize), dtype=np.float32,)
        tmp_size = self.sigma * 3

        for joint_id in range(numJoints):
            # Plus half required due to integer division later
            feat_stride = imageSize / heatmapSize
            mu_x = int(joints[joint_id][0] / feat_stride + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride + 0.5)

            # Check that any part of the gaussian is in-bounds
            if mu_x >= heatmapSize or mu_y >= heatmapSize or mu_x < 0 or mu_y < 0:
                # If not, make joint not visible
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            bl = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            tr = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            # Usable gaussian range
            g_x = max(0, -bl[0]), min(tr[0], heatmapSize) - bl[0]
            g_y = max(0, -bl[1]), min(tr[1], heatmapSize) - bl[1]
            # Image range
            img_x = max(0, bl[0]), min(tr[0], heatmapSize)
            img_y = max(0, bl[1]), min(tr[1], heatmapSize)

            # Only add visible joints (on image) are added to target
            # This occluded joints are visible in this context
            if target_weight[joint_id] > 0:
                target[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
                    g_y[0] : g_y[1], g_x[0] : g_x[1]
                ]
        return target, target_weight

    def calcVisJoints(self, joints2D, shape):
        # Some data sets set non vis joints to negative numbers, these are filtered here
        visJoints = np.where(joints2D >= 0, 1, 0)
        for i in range(len(joints2D)):
            if joints2D[i, 0] >= shape[1]:
                visJoints[i, 0] = 0
            if joints2D[i, 1] >= shape[0]:
                visJoints[i, 1] = 0
        visJoints = visJoints[:, 0] * visJoints[:, 1]
        return visJoints

    def visualiseSample(self, sample, fname):
        image, target, meta = sample
        plt.figure()
        plt.title(meta["imagePath"])
        vis.plotHeatmap(plt, target)   
        vis.plotImage(plt, image)
        plt.savefig(fname)
        

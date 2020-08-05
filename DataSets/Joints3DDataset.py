import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
import torch
import sys
import copy
from PIL import Image

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.JointsDataset import JointsDataset
import DataSets.visualization as vis


class Joints3DDataset(JointsDataset):
    def __init__(self, mode, numJoints, pelvisIndex, connectedJoints, jointColours):
        JointsDataset.__init__(self, mode, numJoints)
        self.pelvisIndex = pelvisIndex
        self.connectedJoints = connectedJoints
        self.jointColours = jointColours

    def _get_db(self):
        pass

    def generateSample(joint2D, joint3D, imagePath, PCKhThreshold):
        return {
            "joints2D": joint2D,
            "joints3D": joint3D,
            "imagePath": imagePath,
            "3DPCKhThreshold": PCKhThreshold
        }

    @abstractmethod
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        anno = copy.deepcopy(self.db[idx])

        meta = {
            "imagePath": anno["imagePath"],
            "3DPCKhThreshold": anno["3DPCKhThreshold"]
        }
        source = torch.tensor(anno["joints2D"], dtype=torch.float32)
        target = torch.tensor(anno["joints3D"], dtype=torch.float32)
        # 3D joints are zeroed off the pelvis joint
        target = target - target[self.pelvisIndex]
        return source.view(-1), target.view(-1), meta

    def visualiseSample(self, sample, fname):
        joints2D, joints3D, meta = sample

        plt.figure()
        ax = plt.subplot(1, 2, 1)
        ax.imshow(Image.open(meta["imagePath"]))

        ax = plt.subplot(1, 2, 2, projection="3d")
        vis.plot3DJoints(ax, joints3D, self.connectedJoints, self.jointColours)

        plt.savefig(fname)

import numpy as np
import matplotlib.pyplot as plt
import os
from abc import abstractmethod
import torch
import sys
import copy
from PIL import Image

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from DataSets.Abstract.JointsDataset import JointsDataset
import DataSets.Utils.Visualisation as vis


class EndToEndDataset(JointsDataset):
    """
        Defines common interface for all dataset used as input to full end to end
        # 3D pose estimation model as well as pre-processing functions for such data.
    """
    def __init__(self, mode, numJoints, pelvisIndex, connectedJoints, jointColours):
        JointsDataset.__init__(self, mode, numJoints)
        self.pelvisIndex = pelvisIndex
        self.connectedJoints = connectedJoints
        self.jointColours = jointColours

    def _get_db(self):
        pass

    def generateSample(
        joints2D, joints3D, imagePath, scale, centre, PCKh2DThreshold, PCKh3DThreshold
    ):
        return {
            "joints2D": joints2D,
            "joints3D": joints3D,
            "imagePath": imagePath,
            "scale": scale,
            "centre": centre,
            "2DPCKhThreshold": PCKh2DThreshold,
            "3DPCKhThreshold": PCKh3DThreshold,
        }

    @abstractmethod
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        anno = copy.deepcopy(self.db[idx])

        image = np.array(Image.open(anno["imagePath"]))
        orgImageShape = np.shape(image)

        # If centre and scale not in annotation calculate from 2D joint positions
        if anno["centre"] is None or anno["scale"] is None:
            anno["centre"], anno["scale"] = self.getBboxCentreAndScaleFrom2DJointPos(
                anno["joints2D"], orgImageShape[:-1]
            )

        visJoints = self.calcVisJoints(anno["joints2D"], orgImageShape)

        image, _, _ = self.preProcessImage(image, anno["centre"], anno["scale"])

        meta = {
            "imagePath": anno["imagePath"],
            "joints2D": anno["joints2D"],
            "centre": anno["centre"],
            "scale": anno["scale"],
            "visJoints": visJoints,
            "3DPCKhThreshold": anno["3DPCKhThreshold"],
            "2DPCKhThreshold": anno["2DPCKhThreshold"],
        }

        target = torch.tensor(anno["joints3D"], dtype=torch.float32)
        # 3D joints are zeroed off the pelvis joint
        target = target - target[self.pelvisIndex]

        return image, target.view(-1), meta

    def visualiseSample(self, sample, fname):
        _, joints3D, meta = sample
        plt.figure()
        plt.suptitle(meta["imagePath"])

        ax = plt.subplot(1, 2, 1)
        # Axis 1 vis is input image with 2D ground truth skeleton overlayed
        vis.plotImage(ax, Image.open(meta["imagePath"]))
        vis.plot2DJoints(ax, meta["joints2D"], self.connectedJoints, self.jointColours, meta["visJoints"])

        # Axis 2 visualises the 3D ground truth skeleton
        ax = plt.subplot(1, 2, 2, projection="3d")
        vis.plot3DJoints(ax, joints3D, self.connectedJoints, self.jointColours)

        plt.savefig(fname)

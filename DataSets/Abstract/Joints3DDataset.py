import matplotlib.pyplot as plt
import os
import torch
import sys
import copy
from abc import abstractmethod

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from DataSets.Abstract.JointsDataset import JointsDataset
import DataSets.Utils.Visualisation as vis


class Joints3DDataset(JointsDataset):
    """
        Defines common interface for all dataset used as input to a 3D lifting network as
        well as pre-processing functions for such data.
    """

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
            "3DPCKhThreshold": PCKhThreshold,
        }

    @abstractmethod
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        anno = copy.deepcopy(self.db[idx])

        meta = {
            "imagePath": anno["imagePath"],
            "3DPCKhThreshold": anno["3DPCKhThreshold"],
            "joints2D": anno["joints2D"],
        }

        source = torch.tensor(anno["joints2D"], dtype=torch.float32)
        target = torch.tensor(anno["joints3D"], dtype=torch.float32)

        # 3D joints are zeroed off the pelvis joint
        target = target - target[self.pelvisIndex]

        return source.view(-1), target.view(-1), meta

    def visualiseSample(self, sample, fname):
        joints2D, joints3D, meta = sample
        plt.figure()
        plt.suptitle(meta["imagePath"])
        ax = plt.subplot(1, 2, 1)

        # Plot input 2D joints on axis 1
        vis.plot2DJoints(
            ax, joints2D.reshape(-1, 2), self.connectedJoints, self.jointColours
        )
        ax.invert_yaxis()

        # Plot target 3D ground truths on axis 2
        ax = plt.subplot(1, 2, 2, projection="3d")
        vis.plot3DJoints(ax, joints3D, self.connectedJoints, self.jointColours)

        plt.savefig(fname)

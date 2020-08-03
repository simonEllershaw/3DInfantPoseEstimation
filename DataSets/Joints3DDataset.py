import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
import torch
import sys
import copy

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.JointsDataset import JointsDataset


class Joints3DDataset(JointsDataset):
    def __init__(self, mode, numJoints, pelvisIndex, connectedJoints):
        super().__init__(mode, numJoints)
        self.pelvisIndex = pelvisIndex
        self.connectedJoints = connectedJoints

    def _get_db(self):
        pass

    def generateSample(joint2D, joint3D, imagePath):
        return {
            "joints2D": joint2D,
            "joints3D": joint3D,
            "imagePath": imagePath,
        }

    @abstractmethod
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        anno = copy.deepcopy(self.db[idx])

        meta = {
            "imagePath": anno["imagePath"],
        }
        source = torch.tensor(anno["joints2D"], dtype=torch.float32)
        target = torch.tensor(anno["joints3D"], dtype=torch.float32)
        # 3D joints are zeroed off the pelvis joint
        target = target - target[self.pelvisIndex]
        return source.view(-1), target.view(-1), meta

    def visualiseSample(self, sample, ax):
        joints2D, joints3D, meta = sample
        plt.tight_layout()
        ax.set_title(meta["imagePath"])

        if torch.is_tensor(joints3D):
            joint3D = joints3D.cpu().numpy()
        joints3D = joints3D.reshape(-1, 3)
        for i in np.arange(len(self.connectedJoints)):
            x, y, z = [
                np.array(
                    [
                        joints3D[self.connectedJoints[i, 0], j],
                        joints3D[self.connectedJoints[i, 1], j],
                    ]
                )
                for j in range(3)
            ]
            ax.plot(x, y, z, lw=2, c="red")

        ax.scatter(joints3D[:, 0], joints3D[:, 1], joints3D[:, 2], c="black")

        # RADIUS = 0.25  # space around the subject
        # xroot, yroot = joints3D[0, 0], joints3D[0, 1]
        # zmax = np.amax(joints3D, axis=0)[2]
        # ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
        # ax.set_zlim3d([-RADIUS + 2 * zmax, zmax])
        # ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
        # ax.set_xlim3d([0, 480])
        # ax.set_ylim3d([0, 640])
        ax.set_xlim3d([-1000, 1000])
        ax.set_ylim3d([-1000, 1000])
        ax.set_xlim3d([-1000, 1000])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        elev = -70
        azim = -90
        ax.view_init(elev, azim)

        # ax.axis("off")
        # if torch.is_tensor(image):
        #     image = image.permute(1, 2, 0).cpu().numpy()
        # ax.imshow(image)

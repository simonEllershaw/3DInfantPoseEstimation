import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import copy
from abc import abstractmethod
import torch.nn.functional as nnf

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.Joints2DDataset import Joints2DDataset
from DataSets.TargetType import TargetType
import DataSets.config as cfg


class MPIIDataset(Joints2DDataset):
    """MPII dataset http://human-pose.mpi-inf.mpg.de/#overview
        Adpated from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""

    def __init__(self, mode):
        # Preprocessing constants
        Joints2DDataset.__init__(self, mode, cfg.MPII["numJoints"])

        # Load setup relevant datasets
        self.datasets = cfg.MPII["modeDatasets"][mode]
        self.annotationFileDirectory = cfg.MPII["annotationFileDirectory"]
        self.imageDirectory = cfg.MPII["imageDirectory"]
        self.db = self._get_db()

    @abstractmethod
    def _get_db(self):
        db = []
        for dataset in self.datasets:
            fileName = os.path.join(self.annotationFileDirectory, dataset + ".json")
            with open(fileName, "r") as file:
                annos = json.load(file)
                for anno in annos:
                    imagePath = os.path.join(self.imageDirectory, anno["image"])
                    joints2D = np.array(anno["joints"])
                    centre = np.array(anno["center"])
                    scale = anno["scale"]
                    db.append(
                        Joints2DDataset.generateSample(
                            joints2D, imagePath, scale, centre
                        )
                    )
        return db

    def getDataLoader(batchSize):
        MPIIData = {x: MPIIDataset(x) for x in ["train", "val"]}
        num_workers = 4
        dataloaders = {
            "train": torch.utils.data.DataLoader(
                MPIIData["train"],
                batch_size=batchSize,
                shuffle=True,
                num_workers=num_workers,
            ),
            "val": torch.utils.data.DataLoader(
                MPIIData["val"],
                batch_size=batchSize,
                shuffle=False,
                num_workers=num_workers,
            ),
        }
        return dataloaders, cfg.MPII["numJoints"]


if __name__ == "__main__":
    data2D = MPIIDataset("train")
    sample = data2D[1457]
    print(sample[1])
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    data2D.visualiseSample(sample, ax)

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    plt.savefig(os.path.join(__location__, "../images/MPII.png"))

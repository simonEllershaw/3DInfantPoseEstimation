import os
import sys
import numpy as np
import json
import torch
from abc import abstractmethod

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from DataSets.Abstract.Joints2DDataset import Joints2DDataset
import DataSets.Utils.Config as cfg


class MPIIDataset(Joints2DDataset):
    """
        Class to load MPII dataset http://human-pose.mpi-inf.mpg.de/#overview
        Used for pretraing of 2D pose estimation
    """

    def __init__(self, mode):
        Joints2DDataset.__init__(self, mode, cfg.MPII["numJoints"])
        self.annotationFileDirectory = cfg.MPII["annotationFileDirectory"]
        self.imageDirectory = cfg.MPII["imageDirectory"]
        # Define dataset split depending on mode
        self.datasets = cfg.MPII["modeDatasets"][mode]
        self.neckJoint = cfg.MPII["neckIndex"]
        self.headJoint = cfg.MPII["headIndex"]
        self.db = self._get_db()

    @abstractmethod
    def _get_db(self):
        db = []
        # Iterate through annotations for each dataset
        for dataset in self.datasets:
            fileName = os.path.join(self.annotationFileDirectory, dataset + ".json")
            with open(fileName, "r") as file:
                annos = json.load(file)
                for anno in annos:
                    imagePath = os.path.join(self.imageDirectory, anno["image"])

                    joints2D = np.array(anno["joints"])
                    # Do not add corrupted (all fill values) samples to db
                    if np.all(joints2D == -1):
                        continue

                    # PCKh value given by euclidean distance between head and neck
                    PCKhThreshold = np.sqrt(
                        np.sum(np.square(joints2D[self.neckJoint] - joints2D[self.headJoint]))
                    )

                    centre = np.array(anno["center"])
                    scale = anno["scale"]

                    db.append(
                        Joints2DDataset.generateSample(
                            joints2D, imagePath, scale, centre, PCKhThreshold
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
        return dataloaders


if __name__ == "__main__":
    # Test function to load dataset and visualise a sample
    data2D = MPIIDataset("train")
    print(len(data2D))
    sample = data2D[4]
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    fname = os.path.join(__location__, "../../Images/MPII.png")
    data2D.visualiseSample(sample, fname)

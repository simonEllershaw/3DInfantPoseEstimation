import scipy.io as sio
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
import sys
import torch
import pickle
import numpy as np
from PIL import Image

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.TargetType import TargetType
from DataSets.Joints3DDataset import Joints3DDataset
import DataSets.config as cfg
import DataSets.visualization as vis

class MPI_INF_3DHPDataset(Joints3DDataset):
    """MPI-INF-3DHP dataset."""

    def __init__(self, mode, targetType, generateAnnotationsJSON=False):
        if targetType is TargetType.joint3D:
            Joints3DDataset.__init__(
                self,
                mode,
                cfg.Hesse["numJoints"],
                cfg.Hesse["pelvicIndex"],
                cfg.Hesse["connectedJoints"],
                cfg.Hesse["jointColours"]
            )
        self.basePath = cfg.MPI_INF["basePath"]
        self.subjects = cfg.MPI_INF["modeSubjects"][mode]
        self.sequences = cfg.MPI_INF["sequences"]
        self.cameras = cfg.MPI_INF["cameras"]
        self.numFrames = cfg.MPI_INF["numFrames"]
        self.annotionsFname = cfg.MPI_INF["annotionsFname"]
        self.hesseMappings = cfg.MPI_INF["hesseMappings"]
        self.jointsNotInHesse = cfg.MPI_INF["jointsNotInHesse"]
        self.PCKhThreshold = cfg.MPI_INF["PCKhThreshold"]

        if generateAnnotationsJSON:
            self.generateAnnotationsJSON()
        self.db = self._get_db()

    # def getRGBImagePath(self, seqDirectory, camera, frameNumber):
    #     return os.path.join(
    #         seqDirectory, "imageSequence", f"{camera:02}", f"frame{frameNumber:05}.jpg"
    #     )

    def getSequenceDirectory(self, subject, sequence):
        return os.path.join(self.basePath, subject, sequence)

    # def getMaskImagePath(self, seqDirectory, camera, frameNumber):
    #     return os.path.join(
    #         seqDirectory, "FGmasks", f"{camera:02}", f"frame{frameNumber:05}.jpg"
    #     )

    @abstractmethod
    def _get_db(self):
        db = []
        for subject in self.subjects:
            for sequence in self.sequences:
                print(subject, sequence)
                seqDirectory = self.getSequenceDirectory(subject, sequence)
                numOfFrames = self.numFrames[subject][sequence]
                # numOfFrames = 5
                annoMatFile = sio.loadmat(os.path.join(seqDirectory, "annot.mat"))
                for frameNumber in range(numOfFrames):
                    for camera in self.cameras:
                        imagePath = self.getRGBImagePath(
                            seqDirectory, camera, frameNumber
                        )

                        joint2D = annoMatFile["annot2"][camera][0][
                            frameNumber
                        ].reshape(-1, 2)
                        joint2D = self.arrayToHesseFormat(joint2D)

                        joint3D = (
                            annoMatFile["annot3"][camera][0][frameNumber]
                            .reshape(-1, 3)
                            .astype("float32")
                        )
                        joint3D = self.arrayToHesseFormat(joint3D)

                        db.append(
                            Joints3DDataset.generateSample(joint2D, joint3D, imagePath, self.PCKhThreshold)
                        )
        return db

    def arrayToHesseFormat(self, jointArray):
        copyOfArray = jointArray.copy()
        for i in range(len(self.hesseMappings)):
            jointArray[i] = copyOfArray[self.hesseMappings[i]]
        return jointArray[:self.numJoints]

    # def generateAnnotationsJSON(self):
    #     db = {}
    #     for subject in self.subjects:
    #         subjectDict = {}
    #         for sequence in self.sequences:
    #             print(subject, sequence)
    #             seqDirectory = self.getSequenceDirectory(subject, sequence)
    #             numOfFrames = self.numFrames[subject][sequence]
    #             # numOfFrames = 5
    #             annoMatFile = sio.loadmat(os.path.join(seqDirectory, "annot.mat"))
    #             sequenceList = []
    #             for frameNumber in range(numOfFrames):
    #                 cameraDict = {}
    #                 for camera in self.cameras:
    #                     joint2D = target = annoMatFile["annot2"][camera][0][
    #                         frameNumber
    #                     ].reshape(-1, 2)

    #                     joint3D = (
    #                         annoMatFile["annot3"][camera][0][frameNumber]
    #                         .reshape(-1, 3)
    #                         .astype("float32")
    #                     )
    #                     sample = {"joints 2D": joint2D, "joints3D": joint3D}
    #                     cameraDict[f"{camera}"] = (sample)
    #             sequenceList.append(cameraDict)
    #         subjectDict[f"{sequence}"] = sequenceList
    #     db[f"{subject}"] = subjectDict

    #     dbfile = open(self.annotionsFname, "wb")
    #     pickle.dump(db, dbfile)
    #     dbfile.close()

    def getDataLoader(batchSize, targetType):
        MPI_INFData = {x: MPI_INF_3DHPDataset(x, targetType) for x in ["train", "val"]}

        num_workers = 12

        dataloaders = {
            "train": torch.utils.data.DataLoader(
                MPI_INFData["train"],
                batch_size=batchSize,
                shuffle=True,
                num_workers=num_workers,
            ),
            "val": torch.utils.data.DataLoader(
                MPI_INFData["val"],
                batch_size=batchSize,
                shuffle=False,
                num_workers=num_workers,
            ),
        }

        return dataloaders, cfg.MPI_INF["numJoints"]


if __name__ == "__main__":
    data = MPI_INF_3DHPDataset("val", TargetType.joint3D)  # , True)
    # print(data[0][0])
    # print(data[0][1])
    # print(len(data))
    # data.loadImages()

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    data.visualiseSample(data[45268], os.path.join(__location__, "../images/MPI_INF_3DHP.png"))
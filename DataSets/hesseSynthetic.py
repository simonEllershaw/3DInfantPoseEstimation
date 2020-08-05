import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from torchvision import transforms
from abc import ABC, abstractmethod
import pickle

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.Joints2DDataset import Joints2DDataset
from DataSets.Joints3DDataset import Joints3DDataset
from DataSets.EndToEndDataset import EndToEndDataset
from DataSets.BboxDataset import BboxDataset
from DataSets.TargetType import TargetType
import DataSets.config as cfg
import DataSets.visualization as vis


class HesseSyntheticDataset(
    Joints3DDataset, Joints2DDataset, BboxDataset, EndToEndDataset
):
    """Hesse Synthetic dataset.
        Adpated from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""

    def __init__(self, mode, targetType, generateBBoxData=False):
        self.targetType = targetType
        if self.targetType == TargetType.bbox:
            self.base = BboxDataset
            BboxDataset.__init__(self, mode, cfg.Hesse["numJoints"])
        elif self.targetType == TargetType.joint2D:
            self.base = Joints2DDataset
            Joints2DDataset.__init__(self, mode, cfg.Hesse["numJoints"])
        elif self.targetType == TargetType.joint3D:
            self.base = Joints3DDataset
            Joints3DDataset.__init__(
                self,
                mode,
                cfg.Hesse["numJoints"],
                cfg.Hesse["pelvicIndex"],
                cfg.Hesse["connectedJoints"],
                cfg.Hesse["jointColours"],
            )
        elif self.targetType == TargetType.endToEnd:
            self.base = EndToEndDataset
            EndToEndDataset.__init__(
                self,
                mode,
                cfg.Hesse["numJoints"],
                cfg.Hesse["pelvicIndex"],
                cfg.Hesse["connectedJoints"],
                cfg.Hesse["jointColours"],
            )

        self.basePath = cfg.Hesse["basePath"]
        self.datasets = cfg.Hesse["modeDatasets"][mode]
        self.connectedJoints = cfg.Hesse["connectedJoints"]
        self.cameraIntrinsics = cfg.Hesse["cameraIntrinsics"]
        self.numFramesPerSequence = cfg.Hesse["numFramesPerSequence"]
        self.videoPCKhThresholds = cfg.Hesse["videoPCKhThresholds"]

        self.boundingBoxPickleFile = os.path.join(self.basePath, "bboxData")
        self.db = self._get_db()

        # This is broken
        # if generateBBoxData:
        # self.generateBoundingBoxPickle()

    @abstractmethod
    def _get_db(self):
        db = []

        if (
            self.targetType == TargetType.joint2D
            or self.targetType == TargetType.endToEnd
        ):
            with open(self.boundingBoxPickleFile, "rb") as bboxDataFile:
                bboxData = pickle.load(bboxDataFile)

        for videoNumber in self.datasets:
            videoPath = self.getVideoPath(videoNumber)
            rbgDirectory = os.path.join(videoPath, "rgb")
            print(videoNumber)
            for frameNumber in range(self.numFramesPerSequence):
                if self.targetType == TargetType.bbox:
                    sample = self.generateBboxSample(
                        videoPath, videoNumber, frameNumber, bboxData
                    )
                elif self.targetType == TargetType.joint2D:
                    sample = self.generate2DJointSample(
                        videoPath, videoNumber, frameNumber, bboxData
                    )
                elif self.targetType == TargetType.joint3D:
                    sample = self.generate3DJointSample(
                        videoPath, videoNumber, frameNumber
                    )
                elif self.targetType == TargetType.endToEnd:
                    sample = self.generateEndToEndSample(
                        videoPath, videoNumber, frameNumber, bboxData
                    )
                db.append(sample)
        return db

    def __getitem__(self, idx):
        return self.base.__getitem__(self, idx)

    def getVideoPath(self, videoNumber):
        return os.path.join(self.basePath, f"{videoNumber:02}")

    def getImageFname(videoFname, frameNumber):
        return os.path.join(videoFname, f"rgb/syn_{frameNumber:05}.png")

    def get2DJointsFname(videoFname, frameNumber):
        return os.path.join(
            videoFname, "joints_2Ddep", f"syn_joints_2Ddep_{frameNumber:05}.txt"
        )

    @staticmethod
    def get3DJointsFname(videoFname, frameNumber):
        return os.path.join(
            videoFname, "joints_3D", f"syn_joints_3D_{frameNumber:05}.txt"
        )

    def getMaskImagePath(self, videoPath, frameNumber):
        return os.path.join(videoPath, "fg_mask", f"mask_{frameNumber:05}.png")

    def generate2DJointSample(self, videoPath, videoNumber, frameNumber, bboxData):
        imagePath = HesseSyntheticDataset.getImageFname(videoPath, frameNumber)

        joints2DFname = HesseSyntheticDataset.get2DJointsFname(videoPath, frameNumber)
        joints2D = np.loadtxt(joints2DFname, dtype="float")[:, :2]

        # Get frameData from bboxData, videoNumber indexed from 1 not 0
        frameData = bboxData[videoNumber - 1][frameNumber]
        scale = frameData["scale"]
        centre = frameData["centre"]

        PCKhThreshold = self.videoPCKhThresholds[videoNumber]["2D"]

        return Joints2DDataset.generateSample(
            joints2D, imagePath, scale, centre, PCKhThreshold
        )

    def generate3DJointSample(self, videoPath, videoNumber, frameNumber):
        imagePath = HesseSyntheticDataset.getImageFname(videoPath, frameNumber)

        joints2DFname = HesseSyntheticDataset.get2DJointsFname(videoPath, frameNumber)
        joints2D = np.loadtxt(joints2DFname, dtype="float")[:, :2]
        # joints2D = np.append(joints2D, np.zeros((3, 2)), axis=0)

        joints3DFname = self.get3DJointsFname(videoPath, frameNumber)
        joints3D = np.loadtxt(joints3DFname, dtype="float")[:, :3] * 1000
        # joints3D = np.append(joints3D, np.zeros((3, 3)), axis=0)

        PCKhThreshold = self.videoPCKhThresholds[videoNumber]["3D"]

        return Joints3DDataset.generateSample(
            joints2D, joints3D, imagePath, PCKhThreshold
        )

    def generateEndToEndSample(self, videoPath, videoNumber, frameNumber, bboxData):
        imagePath = HesseSyntheticDataset.getImageFname(videoPath, frameNumber)

        joints2DFname = HesseSyntheticDataset.get2DJointsFname(videoPath, frameNumber)
        joints2D = np.loadtxt(joints2DFname, dtype="float")[:, :2]

        joints3DFname = self.get3DJointsFname(videoPath, frameNumber)
        joints3D = np.loadtxt(joints3DFname, dtype="float")[:, :3] * 1000

        # Get frameData from bboxData, videoNumber indexed from 1 not 0
        frameData = bboxData[videoNumber - 1][frameNumber]
        scale = frameData["scale"]
        centre = frameData["centre"]

        PCKh2DThreshold = self.videoPCKhThresholds[videoNumber]["2D"]
        PCKh3DThreshold = self.videoPCKhThresholds[videoNumber]["3D"]

        return EndToEndDataset.generateSample(
            joints2D,
            joints3D,
            imagePath,
            scale,
            centre,
            PCKh2DThreshold,
            PCKh3DThreshold,
        )

    def generateBboxSample(self, videoPath, frameNumber):
        imagePath = self.getImageFname(videoPath, frameNumber)
        maskPath = self.getMaskImagePath(videoPath, frameNumber)
        bboxTarget = {
            "boxes": self.getGroundTruthBoundingBox(maskPath, 5),
            "labels": torch.ones((1), dtype=torch.int64),
        }

        return BboxDataset.generateBboxSample(bboxTarget, imagePath)

    def generateBoundingBoxPickle(self):
        db = []
        for videoNumber in range(1, 13):
            videoData = []
            videoPath = self.getVideoPath(videoNumber)
            rbgDirectory = os.path.join(videoPath, "rgb")
            numOfFrames = len(
                [
                    name
                    for name in os.listdir(rbgDirectory)
                    if os.path.isfile(os.path.join(rbgDirectory, name))
                ]
            )
            print(videoNumber)
            for frameNumber in range(numOfFrames):
                frameData = {}
                imagePath = self.getRGBFname(videoPath, frameNumber)
                with Image.open(imagePath) as image:
                    (
                        frameData["scale"],
                        frameData["centre"],
                    ) = self.boundingBoxModel.getCentreAndScale(np.array(image))
                videoData.append(frameData)
            db.append(videoData)
        dbfile = open(self.boundingBoxPickleFile, "wb")
        pickle.dump(db, dbfile)
        dbfile.close()

    def getDataLoader(batchSize, targetType, onlyLoadTest=False):
        dataSets = ["test"] if onlyLoadTest else ["train", "val", "test"]

        syntheticData = {x: HesseSyntheticDataset(x, targetType) for x in dataSets}

        collate_fn = BboxDataset.collate if targetType is TargetType.bbox else None

        num_workers = 4
        dataloaders = {
            "test": torch.utils.data.DataLoader(
                syntheticData["test"],
                batch_size=batchSize,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
        }

        if not onlyLoadTest:
            dataloaders["val"] = torch.utils.data.DataLoader(
                syntheticData["val"],
                batch_size=batchSize,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

            dataloaders["train"] = torch.utils.data.DataLoader(
                syntheticData["train"],
                batch_size=batchSize,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

        return dataloaders, cfg.Hesse["numJoints"]

    def visualiseSample(self, ax, sample):
        self.base.visualiseSample(self, ax, sample)


if __name__ == "__main__":
    syntheticData = HesseSyntheticDataset("val", TargetType.endToEnd)
    sample = syntheticData[0]
    print(sample[2]["3DPCKhThreshold"])

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    syntheticData.visualiseSample(
        syntheticData[0], (os.path.join(__location__, "../images/Hesse.png"))
    )

    # print(joint3D)

    # joint2D = meta["joints2D"]
    # joint3D = meta["joints3D"]
    # z = joint3D[:, 2].reshape(-1,1)
    # numJoints = np.shape(joint2D)[0]
    # intrinsic = syntheticData.getCameraIntrinsics()

    # transform2D = joint3D / z
    # for i in range(len(transform2D)):
    #     transform2D[i] = np.dot(intrinsic, transform2D[i])
    # # print(transform2D[:,:2] - joint2D)

    # inverse = np.linalg.inv(intrinsic)
    # transform3D = np.concatenate([joint2D, np.ones((numJoints, 1))], axis=1)
    # transform3D = transform3D * z
    # for i in range(len(transform2D)):
    #     transform3D[i] = np.dot(inverse, transform3D[i])
    # print(transform3D - joint3D)


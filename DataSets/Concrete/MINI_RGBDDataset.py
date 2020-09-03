import os
import torch
import numpy as np
from PIL import Image
import sys
from abc import abstractmethod
import pickle

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from DataSets.Abstract.Joints2DDataset import Joints2DDataset
from DataSets.Abstract.Joints3DDataset import Joints3DDataset
from DataSets.Abstract.EndToEndDataset import EndToEndDataset
from DataSets.Abstract.BboxDataset import BboxDataset
from DataSets.Utils.TargetType import TargetType
import DataSets.Utils.Config as cfg


class MINI_RGBDDataset(
    Joints3DDataset, Joints2DDataset, BboxDataset, EndToEndDataset
):
    """
        Class to load MINI-RGBD dataset: https://www.iosb.fraunhofer.de/servlet/is/82920/
        Can inherit from all 4 dataset types
    """

    def __init__(self, mode, targetType, generateBBoxData=False):
        self.targetType = targetType
        # Inheritance dependent on targetType
        if self.targetType == TargetType.bbox:
            self.base = BboxDataset
            BboxDataset.__init__(self, mode, cfg.MINI_RGBD["numJoints"])
        elif self.targetType == TargetType.joint2D:
            self.base = Joints2DDataset
            Joints2DDataset.__init__(self, mode, cfg.MINI_RGBD["numJoints"])
        elif self.targetType == TargetType.joint3D:
            self.base = Joints3DDataset
            Joints3DDataset.__init__(
                self,
                mode,
                cfg.MPII["numJoints"],
                cfg.MPII["pelvicIndex"],
                cfg.MPII["connectedJoints"],
                cfg.MPII["jointColours"],
            )
        elif self.targetType == TargetType.endToEnd:
            self.base = EndToEndDataset
            EndToEndDataset.__init__(
                self,
                mode,
                cfg.MPII["numJoints"],
                cfg.MPII["pelvicIndex"],
                cfg.MPII["connectedJoints"],
                cfg.MPII["jointColours"],
            )

        self.basePath = cfg.MINI_RGBD["basePath"]
        self.datasets = cfg.MINI_RGBD["modeDatasets"][mode]
        self.cameraIntrinsics = cfg.MINI_RGBD["cameraIntrinsics"]
        self.numFramesPerSequence = cfg.MINI_RGBD["numFramesPerSequence"]
        self.videoPCKhThresholds = cfg.MINI_RGBD["videoPCKhThresholds"]
        self.MPIIMapping = cfg.MINI_RGBD["MPIIMapping"]

        self.boundingBoxPickleFile = os.path.join(self.basePath, "bboxData")
        self.db = self._get_db()

        if generateBBoxData:
            self.generateBoundingBoxPickle()

    @abstractmethod
    def _get_db(self):
        db = []

        # Need to crop image according to pre generated Faster R-CNN
        # infant bbox model
        if (
            self.targetType == TargetType.joint2D
            or self.targetType == TargetType.endToEnd
        ):
            with open(self.boundingBoxPickleFile, "rb") as bboxDataFile:
                bboxData = pickle.load(bboxDataFile)

        # Each frame in video added to db
        # Sample contents depends on base dataset
        for videoNumber in self.datasets:
            videoPath = self.getVideoPath(videoNumber)
            print(videoNumber)
            for frameNumber in range(self.numFramesPerSequence):
                if self.targetType == TargetType.bbox:
                    sample = self.generateBboxSample(videoPath, frameNumber)
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

    def get3DJointsFname(videoFname, frameNumber):
        return os.path.join(
            videoFname, "joints_3D", f"syn_joints_3D_{frameNumber:05}.txt"
        )

    def getMaskImagePath(self, videoPath, frameNumber):
        return os.path.join(videoPath, "fg_mask", f"mask_{frameNumber:05}.png")

    def generateBboxSample(self, videoPath, frameNumber):
        imagePath = MINI_RGBDDataset.getImageFname(videoPath, frameNumber)
        maskPath = self.getMaskImagePath(videoPath, frameNumber)
        bboxTarget = {
            "boxes": self.getGroundTruthBoundingBox(maskPath, 5),
            "labels": torch.ones((1), dtype=torch.int64),
        }

        return BboxDataset.generateSample(bboxTarget, imagePath)

    def generate2DJointSample(self, videoPath, videoNumber, frameNumber, bboxData):
        imagePath = MINI_RGBDDataset.getImageFname(videoPath, frameNumber)

        joints2DFname = MINI_RGBDDataset.get2DJointsFname(videoPath, frameNumber)
        joints2D = np.loadtxt(joints2DFname, dtype="float")[:, :2]
        joints2D = self.jointsToMPIIFormat(joints2D)

        # Get frameData from bboxData, videoNumber indexed from 1 not 0
        frameData = bboxData[videoNumber - 1][frameNumber]
        scale = frameData["scale"]
        centre = frameData["centre"]

        PCKhThreshold = self.videoPCKhThresholds[videoNumber]["2D"]

        return Joints2DDataset.generateSample(
            joints2D, imagePath, scale, centre, PCKhThreshold
        )

    def generate3DJointSample(self, videoPath, videoNumber, frameNumber):
        imagePath = MINI_RGBDDataset.getImageFname(videoPath, frameNumber)

        joints2DFname = MINI_RGBDDataset.get2DJointsFname(videoPath, frameNumber)
        joints2D = np.loadtxt(joints2DFname, dtype="float")[:, :2]
        joints2D = self.jointsToMPIIFormat(joints2D)

        joints3DFname = MINI_RGBDDataset.get3DJointsFname(videoPath, frameNumber)
        # Convert joints3D to mm
        joints3D = np.loadtxt(joints3DFname, dtype="float")[:, :3] * 1000
        joints3D = self.jointsToMPIIFormat(joints3D)

        PCKhThreshold = self.videoPCKhThresholds[videoNumber]["3D"]

        return Joints3DDataset.generateSample(
            joints2D, joints3D, imagePath, PCKhThreshold
        )

    def generateEndToEndSample(self, videoPath, videoNumber, frameNumber, bboxData):
        imagePath = MINI_RGBDDataset.getImageFname(videoPath, frameNumber)

        joints2DFname = MINI_RGBDDataset.get2DJointsFname(videoPath, frameNumber)
        joints2D = np.loadtxt(joints2DFname, dtype="float")[:, :2]
        joints2D = self.jointsToMPIIFormat(joints2D)

        joints3DFname = MINI_RGBDDataset.get3DJointsFname(videoPath, frameNumber)
        joints3D = np.loadtxt(joints3DFname, dtype="float")[:, :3] * 1000
        joints3D = self.jointsToMPIIFormat(joints3D)

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

    def generateBoundingBoxPickle(self):
        # Generates and saves db of all boudning boxes for the dataset generated by
        # the Faster-RCNN bounding box model
        db = []
        for videoNumber in range(1, 13):
            videoData = []
            videoPath = self.getVideoPath(videoNumber)
            print(videoNumber)
            for frameNumber in range(self.numFramesPerSequence):
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

        syntheticData = {x: MINI_RGBDDataset(x, targetType) for x in dataSets}

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

        return dataloaders

    def visualiseSample(self, ax, sample):
        self.base.visualiseSample(self, ax, sample)


if __name__ == "__main__":
    # Test function to load dataset and visualise a sample
    syntheticData = MINI_RGBDDataset("test", TargetType.endToEnd)
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    syntheticData.visualiseSample(
        syntheticData[0], (os.path.join(__location__, "../../Images/Hesse.png"))
    )

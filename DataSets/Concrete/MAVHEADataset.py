import os
import torch
import numpy as np
import re
import sys
import json

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from DataSets.Abstract.Joints2DDataset import Joints2DDataset
import DataSets.Utils.Config as cfg


class MAVHEADataset(Joints2DDataset):
    """
        Class to load the MAVHEA real dataset
    """

    def __init__(self):
        Joints2DDataset.__init__(self, "test", cfg.MPII["numJoints"])

        self.baseDirectory = cfg.MAHVEA["baseDirectory"]
        self.labelFname = cfg.MAHVEA["labelsFname"]
        self.MPIIMapping = cfg.MAHVEA["MPIIMapping"]
        self.nonMappedJoints = cfg.MAHVEA["nonMappedJoints"]

        self.db = self._get_db()

    def _get_db(self):
        db = []

        with open(self.labelFname) as f:
            data = json.load(f)

        # Order data list by file then frameNumber
        data = sorted(data, key=MAVHEADataset.getIntegerRepOfFileAndFrameNumber)

        # Get list of videoDirectories
        videoDirectories = [d for (_, d, _) in os.walk(self.baseDirectory)][0]
        sampleNumber = 0

        # Add images for each video to db
        for videoDirectory in videoDirectories:
            videoPath = os.path.join(self.baseDirectory, videoDirectory)
            # Iterate through images in ascenfding order
            for imageFile in sorted(
                os.listdir(videoPath), key=MAVHEADataset.getImageFrameNumber
            ):
                sampleData = data[sampleNumber]
                sampleNumber += 1
                imagePath = os.path.join(videoPath, imageFile)

                # Use ground truth bounding box for crop
                bbox = np.array(sampleData["bounding_box"]["__ndarray__"]).flatten()
                bboxDimensions = bbox[2:] - bbox[:2]
                scale = max(bboxDimensions) / 200  # scale in relation to 200px
                centre = (bbox[2:] + bbox[:2]) / 2

                joints2D = []
                for joint in sampleData["joints"]:
                    joints2D.append(sampleData["joints"][joint]["__ndarray__"])
                joints2D = np.array(joints2D)
                # Samples with not all joints labelled are not added
                if len(joints2D) != cfg.MAHVEA["numJoints"]:
                    continue

                # To MPII format
                joints2D = self.jointsToMPIIFormat(joints2D)
                visJoints = np.ones(self.numJoints)
                # Keypoints for which are not labelled made non vis
                # and set to (0,0)
                visJoints[self.nonMappedJoints] = 0
                joints2D[self.nonMappedJoints] = [0, 0]

                sample = Joints2DDataset.generateSample(
                    joints2D, imagePath, scale, centre, 1, visJoints
                )
                db.append(sample)
        return db

    # https://stackoverflow.com/questions/17336943/removing-non-numeric-characters-from-a-string
    def getImageFrameNumber(string):
        return int(re.sub("[^0-9]", "", string))

    def getIntegerRepOfFileAndFrameNumber(label):
        # Takes label and returns int of format to allow ordering of labels
        # E.g. subject_ID = 1 frame_id = 15 -> 100015
        subject_ID = int(re.sub("[^0-9]", "", label["subject_ID"]))
        frame_id = int(re.sub("[^0-9]", "", label["frame_id"]))
        return subject_ID * pow(10, 5) + frame_id

    def getDataLoader(batchSize):
        num_workers = 4
        dataloaders = torch.utils.data.DataLoader(
            MAVHEADataset(), batch_size=batchSize, shuffle=False, num_workers=num_workers,
        )
        return dataloaders


if __name__ == "__main__":
    # Test function to load and visualise dataset
    data = MAVHEADataset()
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    data.visualiseSample(
        data[0], (os.path.join(__location__, "../../Images/MAVHEA.png"))
    )

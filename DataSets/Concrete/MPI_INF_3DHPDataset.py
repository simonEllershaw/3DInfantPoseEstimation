import scipy.io as sio
import os
import sys
import torch
from abc import abstractmethod

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from DataSets.Utils.TargetType import TargetType
from DataSets.Abstract.Joints3DDataset import Joints3DDataset
import DataSets.Utils.Config as cfg


class MPI_INF_3DHPDataset(Joints3DDataset):
    """
        Class to load MPI-INF-3DHP dataset: http://gvv.mpi-inf.mpg.de/3dhp-dataset/
        Used for pre training of 3D lifting network
    """

    def __init__(self, mode, targetType):
        if targetType is TargetType.joint3D:
            Joints3DDataset.__init__(
                self,
                mode,
                cfg.MPII["numJoints"],
                cfg.MPII["pelvicIndex"],
                cfg.MPII["connectedJoints"],
                cfg.MPII["jointColours"],
            )
        self.basePath = cfg.MPI_INF["basePath"]
        self.subjects = cfg.MPI_INF["modeSubjects"][mode]
        self.sequences = cfg.MPI_INF["sequences"]
        self.cameras = cfg.MPI_INF["cameras"]
        self.numFrames = cfg.MPI_INF["numFrames"]
        self.PCKhThreshold = cfg.MPI_INF["PCKhThreshold"]
        self.MPIIMapping = cfg.MPI_INF["MPIIMapping"]
        self.jointsNotInHesse = cfg.MPI_INF["jointsNotInHesse"]

        self.db = self._get_db()

    def getSequenceDirectory(self, subject, sequence):
        return os.path.join(self.basePath, subject, sequence)

    @abstractmethod
    def _get_db(self):
        db = []
        # Iterate through each subject, sequence and frame and add to db
        for subject in self.subjects:
            for sequence in self.sequences:
                print(subject, sequence)
                seqDirectory = self.getSequenceDirectory(subject, sequence)
                numOfFrames = self.numFrames[subject][sequence]
                annoMatFile = sio.loadmat(os.path.join(seqDirectory, "annot.mat"))
                for frameNumber in range(numOfFrames):
                    for camera in self.cameras:
                        joint2D = annoMatFile["annot2"][camera][0][frameNumber].reshape(
                            -1, 2
                        )
                        joint2D = self.jointsToMPIIFormat(joint2D)

                        joint3D = (
                            annoMatFile["annot3"][camera][0][frameNumber]
                            .reshape(-1, 3)
                            .astype("float32")
                        )
                        joint3D = self.jointsToMPIIFormat(joint3D)

                        db.append(
                            Joints3DDataset.generateSample(
                                joint2D, joint3D, "None", self.PCKhThreshold
                            )
                        )
        return db

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

        return dataloaders


if __name__ == "__main__":
    data = MPI_INF_3DHPDataset("val", TargetType.joint3D)
    print(len(data))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    data.visualiseSample(
        data[0], os.path.join(__location__, "../../Images/MPI_INF_3DHP.png")
    )

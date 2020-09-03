
"""
    Generates the models and associated datasets
"""

import torch.nn as nn
from torchvision import models
import os
import sys
import torch.nn.functional as F

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.Concrete.MPIIDataset import MPIIDataset
from DataSets.Concrete.MINI_RGBDDataset import MINI_RGBDDataset
from DataSets.Concrete.MPI_INF_3DHPDataset import MPI_INF_3DHPDataset

from PoseEstimation.Core.JointsMSELoss import JointsMSELoss
from DataSets.Utils.TargetType import TargetType
from PoseEstimation.ModelArchs.LiftingNetwork3D import LiftingNetwork3D, weight_init
import DataSets.Utils.Config as cfg


class Upsampling(nn.Module):
    """
        Stub added to ResNet backbone to form 2D pose estimation model
        proposed by Xiao et al: https://arxiv.org/abs/1804.06208
    """

    def __init__(self, numJoints):
        super(Upsampling, self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(2048, 256, 4, 2, padding=1)
        self.convTrans2 = nn.ConvTranspose2d(256, 256, 4, 2, padding=1)
        self.convTrans3 = nn.ConvTranspose2d(256, 256, 4, 2, padding=1)
        self.convTrans4 = nn.ConvTranspose2d(256, numJoints, 1, 1)

    def forward(self, x):
        x = F.relu(self.convTrans1(x))
        x = F.relu(self.convTrans2(x))
        x = F.relu(self.convTrans3(x))
        x = self.convTrans4(x)
        return x


def load2DPoseEstimationModel(device):
    resNet = models.resnet50(pretrained=True).to(device)
    modules = list(resNet.children())[:-2]
    resNet = nn.Sequential(*modules)
    resNet = nn.Sequential(resNet, Upsampling(cfg.MPII["numJoints"]))
    return resNet.to(device)


def getMPIITrainingObjects(batchSize, device):
    dataLoaders = MPIIDataset.getDataLoader(batchSize)
    model = load2DPoseEstimationModel(device)
    criterion = JointsMSELoss()
    return dataLoaders, model, criterion


def getHesse2DPoseTrainingObjects(batchSize, device, onlyLoadTest=False):
    dataLoaders = MINI_RGBDDataset.getDataLoader(
        batchSize, TargetType.joint2D, onlyLoadTest
    )
    model = load2DPoseEstimationModel(device)
    criterion = JointsMSELoss()
    return dataLoaders, model, criterion


def get3DLiftingNetwork(device):
    model = LiftingNetwork3D(cfg.MPII["numJoints"]).to(device)
    model.apply(weight_init)
    return model


def getMPI_INFLiftingTrainingObjects(batchSize, device):
    dataLoaders = MPI_INF_3DHPDataset.getDataLoader(batchSize, TargetType.joint3D)
    model = get3DLiftingNetwork(device)
    criterion = nn.MSELoss()
    return dataLoaders, model, criterion


def getHesseLiftingTrainingObjects(batchSize, device, onlyLoadTest=False):
    dataLoaders = MINI_RGBDDataset.getDataLoader(
        batchSize, TargetType.joint3D, onlyLoadTest
    )
    model = get3DLiftingNetwork(device)
    criterion = nn.MSELoss()
    return dataLoaders, model, criterion


def getEndToEndHesseModel(batchSize, device, onlyLoadTest=False):
    dataLoaders = MINI_RGBDDataset.getDataLoader(
        batchSize, TargetType.endToEnd, onlyLoadTest
    )
    pose2DModel = load2DPoseEstimationModel(device)
    liftingModel = get3DLiftingNetwork(device)
    return dataLoaders, pose2DModel, liftingModel

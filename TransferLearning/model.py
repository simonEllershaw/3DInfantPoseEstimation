import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import trainer
from datetime import datetime
import os
import sys
import random
import torchsummary
import time
from torch.optim import Adam
import torch.nn.functional as F


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.MPII import MPIIDataset
from DataSets.hesseSynthetic import HesseSyntheticDataset
from DataSets.MPI_INF_3DHP import MPI_INF_3DHPDataset

import TransferLearning.loss as loss
from FasterRCNN.bboxModel import BoundingBoxModel
from DataSets.TargetType import TargetType
from TransferLearning.liftingModel import LinearModel, weight_init
import DataSets.config as cfg
MPIIModelFname = (
    "/homes/sje116/Diss/TransferLearning/savedModels/02_08_13_55/model.tar"
)

MPI_INF_2DPoseModelFname = (
    "/homes/sje116/Diss/TransferLearning/savedModels/MPI_INF_2DPose/model.tar"
)

Hesse2DPoseModelFname = (
    "/homes/sje116/Diss/TransferLearning/savedModels/Hesse2DPose/model.tar"
)


class Upsampling(nn.Module):
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


class Flatten(nn.Module):
    def __init__(self, numJoints):
        super(Flatten, self).__init__()
        self.conv1 = nn.Conv2d(512, 128, 5, 2)
        self.fc = nn.Linear(128 * 2 * 2, numJoints)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def loadMPIIModel(numJoints, device):
    resNet = models.resnet50(pretrained=True)
    modules = list(resNet.children())[:-2]
    resNet = nn.Sequential(*modules)
    resNet = nn.Sequential(resNet, Upsampling(numJoints))
    return resNet.to(device)


def loadHesse2DPoseModel(numJoints, device, loadChildCheckpoint=True):
    model = loadMPIIModel(cfg.MPII["numJoints"], device)
    if loadChildCheckpoint:
        checkpoint = torch.load(MPIIModelFname)
        model.load_state_dict(checkpoint["model_state_dict"])

    stub = list(model.children())[-1]
    stub.convTrans4 = nn.ConvTranspose2d(256, numJoints, 1, 1).to(device)
    return model


# def loadMPI_INFDepthModel(numJoints, device, loadChildCheckpoint=True):
#     model = loadMPI_INF2DPoseModel(28, device, loadChildCheckpoint=False)
#     if loadChildCheckpoint:
#         checkpoint = torch.load(MPI_INF_2DPoseModelFname)
#         model.load_state_dict(checkpoint["model_state_dict"])

#     newStub = Flatten(numJoints).to(device)
#     resnetBackbone = list(model.children())[:-1]
#     resnetBackbone = nn.Sequential(*resnetBackbone)
#     model = nn.Sequential(resnetBackbone, newStub)
#     return model


def loadHesseDepthModel(numJoints, device, loadChildCheckpoint=True):
    model = loadHesse2DPoseModel(numJoints, device, loadChildCheckpoint=False)
    if loadChildCheckpoint:
        checkpoint = torch.load(Hesse2DPoseModelFname)
        model.load_state_dict(checkpoint["model_state_dict"])

    newStub = Flatten(numJoints).to(device)
    resnetBackbone = list(model.children())[:-1]
    resnetBackbone = nn.Sequential(*resnetBackbone)
    model = nn.Sequential(resnetBackbone, newStub)

    return model


def getMPIITrainingObjects(batchSize, device):
    print("MPII")
    dataLoaders, numJoints = MPIIDataset.getDataLoader(batchSize)

    model = loadMPIIModel(numJoints, device)
    # Set up model training paramters
    criterion = loss.JointsMSELoss()

    return dataLoaders, model, criterion


# def getMPI_INF_3DHP_2DPoseTrainingObjects(imageSize, batchSize, device):
#     dataLoaders, numJoints = MPI_INF_3DHPDataset.getDataLoader(
#         batchSize, TargetType.joint2D
#     )

#     model = loadMPI_INF2DPoseModel(numJoints, device)

#     criterion = loss.JointsMSELoss()

#     return dataLoaders, model, criterion


def getHesse2DPoseTrainingObjects(batchSize, device):
    dataLoaders, numJoints = HesseSyntheticDataset.getDataLoader(
        batchSize, TargetType.joint2D
    )

    model = loadHesse2DPoseModel(numJoints, device)

    criterion = loss.JointsMSELoss()

    return dataLoaders, model, criterion


# def getMPI_INF_3DHPDepthModelTrainingObjects(imageSize, batchSize, device):
#     dataLoaders, numJoints = MPI_INF_3DHPDataset.getDataLoader(
#         batchSize, TargetType.joint3D
#     )

#     model = loadMPI_INFDepthModel(numJoints, device)

#     criterion = loss.DepthMSELoss()

#     return dataLoaders, model, criterion


def getDepthModelTrainingObjects(batchSize, device):
    dataLoaders, numJoints = MPI_INF_3DHPDataset.getMPIInfDataLoader(
        batchSize, TargetType.joint3D
    )

    model = LinearModel(numJoints).to(device)
    model.apply(weight_init)

    criterion = nn.MSELoss() #loss.DepthMSELoss()

    return dataLoaders, model, criterion


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    batchSize = 32

    dataLoaders, model, criterion = getMPIITrainingObjects(
        batchSize, device
    )
    # print(torchsummary.summary(model, (3, imageSize, imageSize)))

    paramsToUpdate = model.parameters()
    optimizer = Adam(paramsToUpdate, lr=1e-4)

    checkpoint = torch.load(
        "/homes/sje116/Diss/TransferLearning/savedModels/03_08_15_56/model.tar"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Setup Directory
    dateAndTime = datetime.now().strftime("%d_%m_%H_%M")
    print(dateAndTime)
    directory = os.path.join(os.path.dirname(__file__), f"savedModels/{dateAndTime}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Train
    trainer.train_model(
        model, dataLoaders, device, criterion, optimizer, directory, 40,
    )

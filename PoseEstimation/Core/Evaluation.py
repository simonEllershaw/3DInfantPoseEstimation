"""
    Module can get PCKh and AJPE for a model on a dataset and visualise
    outputs from a model
    See main function for description of use
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

import Inference
from PIL import Image
from PoseEstimation.ModelArchs import ModelGenerator
import DataSets.Utils.Config as cfg
import DataSets.Utils.Visualisation as vis


def getModelMetrics(pose2DModel, liftingModel, dataloader, config, PCKhFactor):

    print("Evaluating model")
    print("-" * 10)

    numJoints = config["numJoints"]

    running2DJointsError = torch.zeros(numJoints)
    running2DPCKh = torch.zeros(numJoints)
    runningPCKh2 = torch.zeros(numJoints)

    euclidean2DDist = []
    running3DJointsError = torch.zeros(numJoints)
    running3DPCKh = torch.zeros(numJoints)
    jointFrequency = torch.zeros(numJoints)
    euclidean3DDist = []
    numSamples = 0

    for source, target, meta in dataloader:
        if pose2DModel:
            jointFrequency += torch.sum(meta["visJoints"], 0)
            outputs = pose2DModel(source.to(device))
            preds = outputs.detach().cpu().numpy()
            predCoords = Inference.postProcessPredictions(
                preds, meta["centre"].numpy(), meta["scale"].numpy(), 64
            )
            gt = meta["joints2D"]

            predCoords = torch.tensor(predCoords)
            expandedVisJoints = meta["visJoints"].unsqueeze(-1).expand_as(gt)

            gt = gt * expandedVisJoints
            predCoords = predCoords * expandedVisJoints

            batchJointError, batchPCKh, batchEuclDist = calcEvalMetrics(
                predCoords, gt, meta["2DPCKhThreshold"], PCKhFactor
            )
            euclidean2DDist.append(batchEuclDist.numpy())
            running2DJointsError += batchJointError
            running2DPCKh += batchPCKh
            source = predCoords.view(-1, numJoints * 2)
        if liftingModel:
            outputs = (
                liftingModel(source.to(device)).detach().cpu().view(-1, numJoints, 3)
            )
            gt = target.view(-1, numJoints, 3)
            batchJointError, batchPCKh, batchEuclDist = calcEvalMetrics(
                outputs, gt, meta["3DPCKhThreshold"], PCKhFactor
            )
            _, batchPCKh2, _ = calcEvalMetrics(outputs, gt, meta["3DPCKhThreshold"], 2)
            runningPCKh2 += batchPCKh2
            euclidean3DDist.append(batchEuclDist.numpy())
            numSamples += outputs.size(0)
            running3DJointsError += batchJointError
            running3DPCKh += batchPCKh

    table = PrettyTable()
    table.add_column("Joint", config["jointNames"])
    if pose2DModel:
        joint2DError = running2DJointsError / jointFrequency
        PCKh2D = running2DPCKh / jointFrequency * 100
        table.add_column("2D Joint Error", joint2DError.numpy())
        table.add_column("2D PCKh", PCKh2D.numpy())
        table.sortby = "2D PCKh"
    if liftingModel:
        joint3DError = running3DJointsError / numSamples
        PCKh3D = running3DPCKh / numSamples * 100
        PCKh2 = runningPCKh2 / numSamples * 100
        table.add_column("3D Joint Error", joint3DError.numpy())
        table.add_column("3D PCKh", PCKh3D.numpy())
        table.add_column("3D PCKh2", PCKh2.numpy())
        table.sortby = "3D PCKh"
    table.reversesort = True
    print(table)
    print()

    if pose2DModel:
        print("Mean 2D Joint Error")
        print(f"{torch.mean(joint2DError).item():.2f} pixels")
        print("Mean 2D PCKh Error")
        print(f"{torch.mean(PCKh2D).item():.2f} %")
        print()

    if liftingModel:
        print("Mean 3D Joint Error")
        print(f"{torch.mean(joint3DError).item():.2f} mm")
        print("Mean 3D PCKh")
        print((PCKh3D))
        print(f"{torch.mean(PCKh3D).item():.2f} %")
        print("Mean 3D PCKh2 Error")
        print(f"{torch.mean(PCKh2).item():.2f} %")


def calcEvalMetrics(preds, gt, PCKhThresholds, PCKhfactor):
    # Euclidean Distance Between Points
    euclideanDist = calcEuclideanDistance(preds, gt)
    # PCKh
    batchPCKh = calcPCKh(euclideanDist, gt, PCKhThresholds, PCKhfactor)
    # Sum over batch
    batchJointError = torch.sum(euclideanDist, 0)
    return batchJointError, batchPCKh, euclideanDist


def calcEuclideanDistance(vector1, vector2):
    # Accepts tensor of size [batchSize, outputSize, 3]
    euclideanDist = torch.square(vector1 - vector2)
    euclideanDist = torch.sum(euclideanDist, 2)
    euclideanDist = torch.sqrt(euclideanDist)
    return euclideanDist


def calcPCKh(euclideanDist, labels, thresholds, factor):
    thresholds = thresholds.unsqueeze(1) * factor
    maskedDistances = euclideanDist[thresholds[:, 0] > 0]
    thresholds = thresholds[thresholds[:, 0] > 0]

    thresholds = thresholds.expand(-1, maskedDistances.size()[1])
    # Mask threshold onto distances
    ones = torch.ones(maskedDistances.size())
    zeros = torch.zeros(maskedDistances.size())
    # Per frame per joint
    PCKhValues = torch.where(maskedDistances <= thresholds, ones, zeros)
    PCKhValues = torch.where(maskedDistances == 0, zeros, PCKhValues)
    # PCKhValues = torch.prod(PCKhValues, 1)
    return torch.sum(PCKhValues, 0)


def visOutput(
    pose2DModel, liftingModel, dataloader, config, batchSize
):
    numJoints = config["numJoints"]
    connectedJoints = config["connectedJoints"]
    jointColours = config["jointColours"]
    idx = 0
    i = 0
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    for source, target, meta in dataloader:
        fname = os.path.join(__location__, f"../../Images/ModelOutput/{i:04d}.png")
        i += 1

        plt.figure()
        plt.tight_layout()
        numRows = 2 if liftingModel else 1

        # Plot 2D Gt
        ax = plt.subplot(numRows, 2, 1)
        ax.set_title("Input")
        imagePath = meta["imagePath"][idx]
        # Images for MPI_INF dataset were not downloaded due to
        # storage constraints
        if not imagePath == "None":
            image = Image.open(meta["imagePath"][idx])
            vis.plotImage(ax, image)
        else:
            ax.invert_yaxis()
        visJoints = meta["visJoints"][idx] if "visJoints" in meta else None
        vis.plot2DJoints(
            ax, meta["joints2D"][idx], connectedJoints, jointColours, visJoints
        )

        if pose2DModel:
            ax = plt.subplot(numRows, 2, 2)
            ax.set_title("Output 2D")
            vis.plotImage(ax, image)
            outputs = pose2DModel(source.to(device))
            preds = outputs.detach().cpu().numpy()
            predCoords = Inference.postProcessPredictions(
                preds, meta["centre"].numpy(), meta["scale"].numpy(), 64
            )
            vis.plot2DJoints(
                ax,
                predCoords[idx],
                connectedJoints,
                jointColours,  # meta["visJoints"][idx],
            )

            source = torch.tensor(predCoords).view(-1, numJoints * 2)

        if liftingModel:
            ax = plt.subplot(numRows, 2, 3, projection="3d")
            ax.set_title("Ground Truth 3D")
            vis.plot3DJoints(ax, target[idx], connectedJoints, jointColours)

            ax = plt.subplot(numRows, 2, 4, projection="3d")
            ax.set_title("Output 3D")
            outputs = (
                liftingModel(source.to(device)).detach().cpu().view(-1, numJoints, 3)
            )
            vis.plot3DJoints(ax, outputs[idx], connectedJoints, jointColours)

        plt.savefig(fname)
        plt.close()


def plotValidationError(fname):
    plt.figure()
    data = np.loadtxt(fname)
    plt.plot(data[:, 1], label="train")
    plt.plot(data[:, 2], label="validation")
    plt.legend()
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/AU")

    plt.tight_layout()
    plt.savefig("Images/valError.png")


if __name__ == "__main__":
    batchSize = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # dataLoader, pose2DModel, _ = ModelGenerator.getMPIITrainingObjects(batchSize, device)

    # dataLoader, pose2DModel, _ = ModelGenerator.getHesse2DPoseTrainingObjects(
    #     batchSize, device, True
    # )

    # dataLoader, liftingModel, _ = ModelGenerator.getMPI_INFLiftingTrainingObjects(
    #     batchSize, device
    # )

    # dataLoader, liftingModel, _ = ModelGenerator.getHesseLiftingTrainingObjects(
    #     batchSize, device, True
    # )

    dataLoader, pose2DModel, liftingModel = ModelGenerator.getEndToEndHesseModel(
        batchSize, device, True
    )

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    MPIIPose2DFname = "../../SavedModels/MPII2DFlip/model.tar"
    MPI_INFLiftingFname = "../../SavedModels/MPI_INFLifting/model.tar"

    MINI_RGBD_Pose2DFname = "../../SavedModels/MINI_RGBD_2D/model.tar"
    MINI_RGBD_LiftingFname = "../../SavedModels/MINI_RGBD_Lift/model.tar"
    MINI_RGBD_FineTuneFname = "../../SavedModels/MINI_RGBD_FineTune/model.tar"

    checkpoint = torch.load(os.path.join(__location__, MINI_RGBD_Pose2DFname))
    pose2DModel.load_state_dict(checkpoint["model_state_dict"])
    pose2DModel.eval()

    checkpoint = torch.load(os.path.join(__location__, MINI_RGBD_FineTuneFname))
    liftingModel.load_state_dict(checkpoint["model_state_dict"])
    liftingModel.eval()

    # getModelMetrics(pose2DModel, liftingModel, dataLoader["test"], cfg.MPII, 1)

    visOutput(
        pose2DModel, liftingModel, dataLoader["test"], cfg.MPII, batchSize,
    )

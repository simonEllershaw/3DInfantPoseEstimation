import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.hesseSynthetic import HesseSyntheticDataset

import TransferLearning.model as model
from DataSets.MPII import MPIIDataset
from DataSets.MPI_INF_3DHP import MPI_INF_3DHPDataset
from DataSets.TargetType import TargetType
import TransferLearning.inference as inference
from PIL import Image
import DataSets.config as cfg
import DataSets.visualization as vis


def evaluateDepthModel(model, dataloader, numJoints, head, neck, device):

    print("Evaluating model on test set")
    print("-" * 10)
    print(len(dataloader))
    runningJointsError = torch.zeros(numJoints)
    runningPCKh = torch.zeros(numJoints)
    i = 0
    euclideanDist = []
    for joints2D, gt, meta in dataloader:
        i += 1
        # if i == 5:
        #     break
        outputs = model(joints2D.to(device)).detach().cpu().view(-1, numJoints, 3)
        gt = gt.view(-1, numJoints, 3)
        # Zero joints that aren't visible
        # expandedVisJoints = meta["visJoints"].unsqueeze(-1).expand_as(gt)
        # gt = gt * expandedVisJoints
        # predCoords = predCoords * expandedVisJoints

        batchJointError, batchPCKh, batchEuclDist = calcEvalMetrics(
            outputs, gt, head, neck
        )
        # test = batchEuclDist > 900
        # if test.any():
        #     print(i)
        #     print(batchEuclDist)
        #     print(meta["imagePath"])
        #     print()
        euclideanDist.append(batchEuclDist.numpy())
        runningJointsError += batchJointError
        runningPCKh += batchPCKh

        # if firstIteration:
        #     fname = batch["imagePath"][0]
        #     visualiseResult(outputs[0], labels[0], fname, ndim)
        #     firstIteration = False
    # plotHistogram(np.array(euclideanDist))
    jointError = runningJointsError / len(dataloader)
    PCKh = runningPCKh / len(dataloader)

    print("Joint Errors:")
    print(jointError)
    print("Mean Joint Error")
    print(f"{torch.mean(jointError).item():.5f} mm")

    print("PCKH per joint:")
    print(PCKh)
    print("Mean PCKh")
    print(f"{torch.mean(PCKh).item() * 100:.2f}%")


def evaluateModel(pose2DModel, liftingModel, dataloader, config):

    print("Evaluating model on test set")
    print("-" * 10)
    print(len(dataloader))

    numJoints = config["numJoints"]
    head = config["headIndex"]
    neck = config["neckIndex"]

    running2DJointsError = torch.zeros(numJoints)
    running2DPCKh = torch.zeros(numJoints)
    euclidean2DDist = []
    running3DJointsError = torch.zeros(numJoints)
    running3DPCKh = torch.zeros(numJoints)
    euclidean3DDist = []
    i = 0

    for source, target, meta in dataloader:
        i += 1
        # if i == 5:
        #     break

        if pose2DModel:
            outputs = pose2DModel(source.to(device))
            preds = outputs.detach().cpu().numpy()
            predCoords = inference.postProcessPredictions(
                preds, meta["centre"].numpy(), meta["scale"].numpy(), 64
            )
            gt = meta["joints2D"]

            predCoords = torch.tensor(predCoords)
            expandedVisJoints = meta["visJoints"].unsqueeze(-1).expand_as(gt)
            gt = gt * expandedVisJoints
            predCoords = predCoords * expandedVisJoints

            batchJointError, batchPCKh, batchEuclDist = calcEvalMetrics(
                predCoords, gt, head, neck
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
                outputs, gt, head, neck
            )
            euclidean3DDist.append(batchEuclDist.numpy())
            running3DJointsError += batchJointError
            running3DPCKh += batchPCKh

    # plotHistogram(np.array(euclideanDist))
    joint2DError = running2DJointsError / len(dataloader)
    PCKh2D = running2DPCKh / len(dataloader)
    joint3DError = running3DJointsError / len(dataloader)
    PCKh3D = running3DPCKh / len(dataloader)

    if pose2DModel:
        print("2D Joint Errors:")
        print(joint2DError)
        print("Mean 2D Joint Error")
        print(f"{torch.mean(joint2DError).item():.2f} pixels")
        print("2D PCKh per joint:")
        print(PCKh2D)
        print("Mean 2D PCKh Error")
        print(f"{torch.mean(PCKh2D).item() * 100:.2f} %")
        print()

    if liftingModel:
        print("3D Joint Errors:")
        print(joint3DError)
        print("Mean 3D Joint Error")
        print(f"{torch.mean(joint3DError).item():.2f} mm")
        print("3D PCKh per joint:")
        print(PCKh3D)
        print("Mean 3D PCKh")
        print(f"{torch.mean(PCKh3D).item() * 100:.2f} %")


def plotHistogram(euclideanDist):
    histData = np.clip(euclideanDist.flatten(), 0, 100)
    plt.figure()
    plt.hist(histData)
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    plt.savefig(os.path.join(__location__, "../images/hist.png"))


def calcEvalMetrics(preds, gt, head, neck):
    # Euclidean Distance Between Points
    euclideanDist = calcEuclideanDistance(preds, gt)
    # PCKh
    batchPCKh = calcPCKh(euclideanDist, gt, head, neck)
    # Sum over batch
    batchJointError = torch.mean(euclideanDist, 0)
    return batchJointError, batchPCKh, euclideanDist


def calcEuclideanDistance(vector1, vector2):
    # Accepts tensor of size [batchSize, outputSize, 3]
    euclideanDist = torch.square(vector1 - vector2)
    euclideanDist = torch.sum(euclideanDist, 2)
    euclideanDist = torch.sqrt(euclideanDist)
    return euclideanDist


def calcPCKh(euclideanDist, labels, head, neck, factor=0.5):
    # Calc PCKh threshold (head to neck length times a factor)
    headPositions = labels[:, head, :].unsqueeze(1)
    neckPositions = labels[:, neck, :].unsqueeze(1)
    thresholds = calcEuclideanDistance(headPositions, neckPositions) * factor

    maskedDistances = euclideanDist[thresholds[:, 0] > 0]
    thresholds = thresholds[thresholds[:, 0] > 0]

    thresholds = thresholds.expand(-1, maskedDistances.size()[1])
    # thresholds = torch.ones(euclideanDist.size()) * 150
    # Mask threshold onto distances
    ones = torch.ones(maskedDistances.size())
    zeros = torch.zeros(maskedDistances.size())
    # Per frame per joint
    PCKhValues = torch.where(maskedDistances <= thresholds, ones, zeros)
    # PCKhValues = torch.prod(PCKhValues, 1)
    return torch.mean(PCKhValues, 0)


# def visualiseResult(model, dataloader, device):
#     i = 0
#     for image, _, meta in dataloader:
#         if i == 309:
#             break
#         i += 1

#     outputs = model(image.to(device))
#     preds = outputs.detach().cpu().numpy()
#     predCoords = inference.postProcessPredictions(
#         preds, meta["centre"].numpy(), meta["scale"].numpy(), 64
#     )[0]
#     gt = meta["joint2D"][0]
#     visJoints = meta["visJoints"][0]
#     print(gt)
#     print(predCoords)
#     print(visJoints)
#     print(gt - predCoords)

#     plt.figure()
#     plt.title(meta["imagePath"][0])
#     ax = plt.subplot(1, 1, 1)
#     for i in range(len(connectedJoints)):
#         joint1 = connectedJoints[i, 0]
#         joint2 = connectedJoints[i, 1]
#         if visJoints[joint1] == 1 and visJoints[joint2] == 1:
#             x, y = [
#                 np.array([gt[connectedJoints[i, 0], j], gt[connectedJoints[i, 1], j]])
#                 for j in range(2)
#             ]
#             ax.plot(x, y, lw=2, c="r")  # self.connectedJoints[i, 2])
#             x, y = [
#                 np.array(
#                     [
#                         predCoords[connectedJoints[i, 0], j],
#                         predCoords[connectedJoints[i, 1], j],
#                     ]
#                 )
#                 for j in range(2)
#             ]
#             ax.plot(x, y, lw=2, c="b")

#     for i in range(len(visJoints)):
#         if visJoints[i] == 1:
#             ax.scatter(gt[i, 0], gt[i, 1], c="red")
#             ax.scatter(predCoords[i, 0], predCoords[i, 1], c="blue")
#     ax.scatter(predCoords[-1, 0], predCoords[-1, 1], c="black")
#     ax.scatter(gt[-1, 0], gt[-1, 1], c="green")
#     # ax.scatter(gt[1, 0], gt[1, 1], c="red")
#     # ax.scatter(gt[4, 0], gt[4, 1], c="green")
#     # ax.scatter(gt[5, 0], gt[5, 1], c="blue")

#     image = np.array(Image.open(meta["imagePath"][0]))
#     ax.imshow(image)
#     __location__ = os.path.realpath(
#         os.path.join(os.getcwd(), os.path.dirname(__file__))
#     )
#     plt.savefig(os.path.join(__location__, "../images/PredictPose.png"))


def visAnOutput(idx, pose2DModel, liftingModel, dataloader, config, batchSize):
    numJoints = config["numJoints"]
    connectedJoints = config["connectedJoints"]
    jointColours = config["jointColours"]

    for source, target, meta in dataloader:
        if idx < batchSize:
            break
        else:
            idx -= batchSize

    plt.figure()
    numColumns = 2 if liftingModel else 1

    # Plot 2D Gt
    ax = plt.subplot(numColumns, 2, 1)
    ax.set_title("Ground Truth 2D")
    image = Image.open(meta["imagePath"][idx])
    vis.plotImage(ax, image)
    vis.plot2DJoints(ax, meta["joints2D"][idx], connectedJoints, jointColours)

    if pose2DModel:
        ax = plt.subplot(numColumns, 2, 2)
        ax.set_title("Output 2D")
        vis.plotImage(ax, image)
        outputs = pose2DModel(source.to(device))
        preds = outputs.detach().cpu().numpy()
        predCoords = inference.postProcessPredictions(
            preds, meta["centre"].numpy(), meta["scale"].numpy(), 64
        )
        vis.plot2DJoints(ax, predCoords[idx], connectedJoints, jointColours)
        source = torch.tensor(predCoords).view(-1, numJoints * 2)

    if liftingModel:
        ax = plt.subplot(numColumns, 2, 3, projection="3d")
        ax.set_title("Ground Truth 3D")
        vis.plot3DJoints(
            ax, target[idx], connectedJoints, jointColours
        )

        ax = plt.subplot(numColumns, 2, 4, projection="3d")
        ax.set_title("Output 3D")
        outputs = liftingModel(source.to(device)).detach().cpu().view(-1, numJoints, 3)
        vis.plot3DJoints(
            ax, outputs[idx], connectedJoints, jointColours
        )

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    plt.savefig(os.path.join(__location__, "../images/Output.png"))


def plotValidationError(fname):
    plt.figure()
    data = np.loadtxt(fname)
    plt.plot(data[:, 1], label="train")
    plt.plot(data[:, 2], label="val")
    plt.legend()
    plt.title(fname)
    plt.savefig("images/valError.png")


if __name__ == "__main__":
    batchSize = 16

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # dataLoader, model, _ = model.getHesseLiftingTrainingObjects(batchSize, device)

    dataLoaders, pose2DModel, liftingModel = model.getEndToEndHesseModel(
        batchSize, device
    )

    hessePose2DFname = (
        "/homes/sje116/Diss/TransferLearning/savedModels/04_08_12_19/model.tar"
    )
    hesseLifitingFname = (
        "/homes/sje116/Diss/TransferLearning/savedModels/05_08_11_34/model.tar"
    )

    checkpoint = torch.load(hessePose2DFname)
    pose2DModel.load_state_dict(checkpoint["model_state_dict"])
    pose2DModel.eval()

    checkpoint = torch.load(hesseLifitingFname)
    liftingModel.load_state_dict(checkpoint["model_state_dict"])
    liftingModel.eval()

    # evaluateModel(pose2DModel, liftingModel, dataLoaders["test"], cfg.Hesse)

    visAnOutput(0, pose2DModel, liftingModel, dataLoaders["test"], cfg.Hesse, batchSize)

    # plotValidationError(
    #     "/homes/sje116/Diss/TransferLearning/savedModels/04_08_12_58/model.tar"
    # )

    # torch.set_printoptions(precision=5)

    # image, targets, _ = next(iter(dataLoader["val"]))
    # output = model(image.to(device))[0]
    # output = output.detach().cpu().numpy()
    # # output = np.clip(output, 0, 1)
    # target = targets[0]
    # for i in range(len(output)):
    #     plt.figure()
    #     ax = plt.subplot(1, 2, 1)
    #     plt.axis("square")
    #     ax = sns.heatmap(output[i])  # / np.max(output[i]))

    #     ax = plt.subplot(1, 2, 2)
    #     plt.axis("square")
    #     ax = sns.heatmap(target[i])
    #     __location__ = os.path.realpath(
    #         os.path.join(os.getcwd(), os.path.dirname(__file__))
    #     )
    #     plt.savefig(os.path.join(__location__, f"../images/MPIIOutputs/{i}.png"))
    #     plt.close()

    # plt.figure()
    # plt.imshow(image.permute(1, 2, 0).numpy())
    # __location__ = os.path.realpath(
    #     os.path.join(os.getcwd(), os.path.dirname(__file__))
    # )
    # plt.savefig(os.path.join(__location__, "../images/MPIIOutputs/image.png"))
    # plt.figure()
    # ax = plt.subplot(1, 1, 1)
    # MPIIDataset.visualiseSample((image, target, meta), ax)
    # __location__ = os.path.realpath(
    #     os.path.join(os.getcwd(), os.path.dirname(__file__))
    # )
    # plt.savefig(os.path.join(__location__, "../images/MPIIOutputs/gt.png"))
    # for layer in output:

    # loadedModel.eval()
    # loadedModel.to("cpu")
    # recontruct = {x: meta[x][0] for x in meta.keys()}
    # plt.figure()
    # ax = plt.subplot(1, 2, 1)
    # MPIIDataset.visualiseSample((image[0], target[0], recontruct), ax)

    # ax = plt.subplot(1, 2, 2)
    # output = loadedModel(image)
    # MPIIDataset.visualiseSample((image[0], output.cpu(), recontruct), ax)

    # __location__ = os.path.realpath(
    #     os.path.join(os.getcwd(), os.path.dirname(__file__))
    # )
    # plt.savefig(os.path.join(__location__, "../images/MPIIeval.png"))

    # evaluateModel(model, data["val"], device, outputSize, ndim=2)
    # plotValidationError()

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
        test = batchEuclDist > 900
        if test.any():
            print(i)
            print(batchEuclDist)
            print(meta["imagePath"])
            print()
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


def evaluateModel(model, dataloader, numJoints, head, neck):

    print("Evaluating model on test set")
    print("-" * 10)
    print(len(dataloader))
    runningJointsError = torch.zeros(numJoints)
    runningPCKh = torch.zeros(numJoints)
    i = 0
    euclideanDist = []
    for image, _, meta in dataloader:
        i += 1
        # if i == 5:
        #     break
        outputs = model(image.to(device))

        preds = outputs.detach().cpu().numpy()
        predCoords = inference.postProcessPredictions(
            preds, meta["centre"].numpy(), meta["scale"].numpy(), 64
        )
        gt = meta["joints2D"]

        predCoords = torch.tensor(predCoords)
        # plt.figure()
        # plt.scatter(gt[:, 0], gt[:, 1], color="red")
        # plt.scatter(predCoords[:, 0], predCoords[:, 1], color="blue")
        # plt.imshow(Image.open(meta["imagePath"][0]))
        # __location__ = os.path.realpath(
        #     os.path.join(os.getcwd(), os.path.dirname(__file__))
        # )
        # plt.savefig(os.path.join(__location__, "eval.png"))
        # Zero joints that aren't visible
        expandedVisJoints = meta["visJoints"].unsqueeze(-1).expand_as(gt)
        gt = gt * expandedVisJoints
        predCoords = predCoords * expandedVisJoints

        batchJointError, batchPCKh, batchEuclDist = calcEvalMetrics(
            predCoords, gt, head, neck
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

    # plotHistogram(np.array(euclideanDist))
    jointError = runningJointsError / len(dataloader)
    PCKh = runningPCKh / len(dataloader)

    print("Joint Errors:")
    print(jointError)
    print("Mean Joint Error")
    print(f"{torch.mean(jointError).item():.2f} pixels")

    print("PCKH per joint:")
    print(PCKh)
    print("Mean PCKh")
    print(f"{torch.mean(PCKh).item() * 100:.2f}%")


def plotHistogram(euclideanDist):
    histData = np.clip(euclideanDist.flatten(), 0, 100)
    plt.figure()
    plt.hist(histData)
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    plt.savefig(os.path.join(__location__, f"../images/hist.png"))


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


def vis3DModelOutput(idx, connectedJoints, model, device, dataloader, batchSize):
    for joints2D, gt, meta in dataloader:
        if idx < batchSize:
            break
        else:
            idx -= batchSize
    outputs = model(joints2D.to(device))

    plt.figure()
    ax = plt.subplot(1, 3, 1)
    image = Image.open(meta["imagePath"][idx])
    print(meta["imagePath"])
    ax.imshow(image)
    ax = plt.subplot(1, 3, 2, projection="3d")
    vis.plot3DJoints(ax, gt[idx], connectedJoints, "gt")
    ax = plt.subplot(1, 3, 3, projection="3d")
    vis.plot3DJoints(ax, outputs[idx], connectedJoints, "output")

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    plt.savefig(os.path.join(__location__, "../images/MPI_INF3DOutput.png"))


def plotValidationError(fname):
    plt.figure()
    data = np.loadtxt(fname)
    plt.plot(data[:, 1], label="train")
    plt.plot(data[:, 2], label="val")
    plt.legend()
    plt.title(fname)
    plt.savefig("images/valError.png")


if __name__ == "__main__":
    dataConfig = cfg.Hesse
    head = dataConfig["headIndex"]
    neck = dataConfig["neckIndex"]
    numJoints = dataConfig["numJoints"]
    connectedJoints = dataConfig["connectedJoints"]
    batchSize = 16

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    dataLoader, model, _ = model.getHesse2DPoseTrainingObjects(batchSize, device)
    # model = model.loadHesseDepthModel(numJoints, device)
    checkpoint = torch.load(
        "/homes/sje116/Diss/TransferLearning/savedModels/03_08_16_28/model.tar"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # vis3DModelOutput(16443, connectedJoints, model, device, dataLoader["val"], batchSize)
    evaluateModel(model, dataLoader["test"], numJoints, head, neck)
    # evaluateDepthModel(model, dataLoader["val"], numJoints, head, neck, device)
    # plotValidationError(
    #     "/homes/sje116/Diss/TransferLearning/savedModels/03_08_15_56/metrics.txt"
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

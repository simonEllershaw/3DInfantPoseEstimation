import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import TransferLearning.model as model
from DataSets.MPII import MPIIDataset
from DataSets.transforms import get_affine_transform, affine_transform
from DataSets.MPI_INF_3DHP import MPI_INF_3DHPDataset
from DataSets.hesseSynthetic import HesseSyntheticDataset
from DataSets.TargetType import TargetType


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(
        batch_heatmaps, np.ndarray
    ), "batch_heatmaps should be numpy.ndarray"
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def getBatchAveragePixelDistance(predict, gt, visJoints):
    batch_size = np.shape(predict)[0]
    num_joints = np.shape(predict)[1]

    avPixelDistance = 0
    for idx in range(num_joints):
        jointPredict = predict[:, idx].squeeze()
        jointGt = gt[:, idx].squeeze()
        # If joint not visible both target and output values put to 0
        jointTargetWeights = visJoints[:, idx].reshape(-1, 1)
        jointPredict = np.multiply(jointTargetWeights, jointPredict)
        jointGt = np.multiply(jointTargetWeights, jointGt)
        pixelDistance = np.sqrt(np.sum(np.square(jointPredict - jointGt), axis=1))
        avPixelDistance += np.sum(pixelDistance)
    return avPixelDistance / np.sum(visJoints)

def postProcessPredictions(preds, centre, scale, cropSize):
    coords = get_max_preds(preds)
    for i in range(coords.shape[0]):
        coords[i] = transform_preds(coords[i], centre[i], scale[i], cropSize)
    return coords

def getAveragePixelDistance(model, data):
    cumulativePixelDistance = 0
    for image, target, meta in data:
        preds = model(image.to(device))
        preds = preds.detach().cpu().numpy()
        predCoords = postProcessPredictions(preds, meta["centre"].numpy(), meta["scale"].numpy(), 64)
        gt = meta["joint2D"].numpy()
        cumulativePixelDistance += getBatchAveragePixelDistance(
            predCoords, gt, meta["visJoints"].numpy()
        )
    numBatches = len(data)
    return cumulativePixelDistance / numBatches


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batchSize = 1

    data, numJoints = dataloaders.getHesseDataLoader(batchSize, 256, TargetType.joint2D)

    model = model.loadMPI_INF2DPoseModel(numJoints, device)
    checkpoint = torch.load(
        "/homes/sje116/Diss/TransferLearning/savedModels/Hesse2DPose/model.tar"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(getAveragePixelDistance(model, data["test"]))


    # avPixelDistance = 0
    # for image, target, meta in hesseDataloader["test"]:
    #     preds = model(image.to(device))
    #     coords, _ = get_max_preds(output)
    #     gt = meta["jointPosOrg"].to(device)
    #     avPixelDistance += getBatchAveragePixelDistance(
    #         preds, gt, meta["visJoints"].to(device)
    #     )
    # numBatches = len(hesseDataloader["test"])
    # print(avPixelDistance / numBatches)

    # data = HesseSyntheticDataset("test", 256, TargetType.joint2D)


#2418
    # image, target, meta = data[100]
    # image = image.unsqueeze(0)
    # target = target.reshape(1, 25, 64, 64)
    # output = model(image.to(device))
    # output = output.detach().cpu().numpy()
    # coords, _ = get_max_preds(output)
    # preds = coords.copy()

    # heatmap_height = target.shape[2]
    # heatmap_width = target.shape[3]

    # gt = meta["jointPosOrg"]

    # for i in range(coords.shape[0]):
    #     preds[i] = transform_preds(
    #         coords[i], meta["centre"], meta["scale"], heatmap_width
    #     )
    
    # plt.figure()
    # plt.tight_layout()
    # ax = plt.subplot(1, 1, 1)
    # ax.set_title(meta["ID"])

    # # # Plot gaussians
    # # levels = np.arange(0.4, 1.2, 0.2)
    # # target = target.view(1, 16, 64, 64)
    # # upsampledJoints = nnf.interpolate(target, size=(256, 256), mode="bilinear")
    # # for joint in upsampledJoints[0]:
    # #     ax.contourf(joint, levels, alpha=0.9)

    # # Plot visible joint ground truths
    # visJoints = meta["visJoints"]
    # for i in range(len(visJoints)):
    #     if visJoints[i] > 0:
    #         ax.scatter(
    #             gt[i, 0], gt[i, 1], s=50, marker=".", c="r",
    #         )
    #         ax.scatter(
    #             preds[0][i, 0], preds[0][i, 1], s=50, marker=".", c="b",
    #         )

    # # ax.axis("off")
    # image = np.array(Image.open(meta["ID"]))
    # ax.imshow(image)

    # __location__ = os.path.realpath(
    #     os.path.join(os.getcwd(), os.path.dirname(__file__))
    # )
    # plt.savefig(os.path.join(__location__, "../images/inference.png"))

# def visualiseOutput(model, data):


# else:  
#             groundTruth = meta["joint2D"]
#             for i in np.arange(len(self.connectedJoints)):
#                 x, y = [
#                     np.array(
#                         [
#                             groundTruth[self.connectedJoints[i, 0] - 1, j],
#                             groundTruth[self.connectedJoints[i, 1] - 1, j],
#                         ]
#                     )
#                     for j in range(2)
#                 ]
#                 ax.plot(x, y, lw=2, c="r")#self.connectedJoints[i, 2])

#             ax.scatter(groundTruth[:, 0], groundTruth[:, 1], c="black")
#             image = np.array(Image.open(meta["imagePath"]))
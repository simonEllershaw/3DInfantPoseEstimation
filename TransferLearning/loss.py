# Copied from https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/loss.py

import torch.nn as nn
import torch

# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import model
# import DataSets.dataloaders as dataloaders


class JointsMSELoss(nn.Module):
    """
        Calculates MSE on 2 heatmaps, ignoring non visible joints
    """

    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="sum")

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            # If joint not visible both target and output values put to 0
            jointTargetWeights = target_weight[:, idx].view(-1, 1)
            heatmap_pred = torch.mul(jointTargetWeights, heatmap_pred)
            heatmap_gt = torch.mul(jointTargetWeights, heatmap_gt)
            loss += self.criterion(heatmap_pred, heatmap_gt)
        # Average over total num predictions made
        numPredictions = torch.sum(target_weight) * target.size(2) * target.size(3)
        return loss / numPredictions


class DepthMSELoss(nn.Module):
    """
        Calculates MSE on 2 heatmaps, ignoring non visible joints
    """

    def __init__(self):
        super(DepthMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="sum")

    def forward(self, output, target, target_weight):
        outputVisMasked = output * target_weight
        targetVisMasked = target * target_weight
        loss = self.criterion(outputVisMasked, targetVisMasked)
        # Average over total num predictions made
        numPredictions = torch.sum(target_weight)
        return loss / numPredictions


# if __name__ == "__main__":
#     dataLoaders, numJoints = dataloaders.getMPIIDataLoader(1, 256)
#     image, target, meta = next(iter(dataLoaders["train"]))
#     device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     MPIImodel = model.loadModel(numJoints, device)
#     checkpoint = torch.load("/homes/sje116/Diss/TransferLearning/savedModels/07_07_11_02/model.tar")

#     MPIImodel.load_state_dict(checkpoint["model_state_dict"])
#     output = MPIImodel(image.to(device))
#     loss = JointsMSELoss()
#     calcLoss = loss(output, target, meta["visJoints"])
#     print(calcLoss)

#     plt.figure()
#     ax = plt.subplot(1, 3, 1)
#     plt.axis('square')
#     ax = sns.heatmap(target[0][2])

#     ax = plt.subplot(1, 3, 2)
#     plt.axis('square')
#     ax = sns.heatmap(output[0][2].detach())

#     ax = plt.subplot(1, 3, 3)
#     ax.imshow(image[0].permute(1, 2, 0).numpy())

#     __location__ = os.path.realpath(
#         os.path.join(os.getcwd(), os.path.dirname(__file__))
#     )
#     plt.savefig(os.path.join(__location__, "../images/MPIIOutputs/test.png"))

# target = torch.ones((2, 2, 1, 4))

#     # output = torch.ones((2, 2, 1, 4))
#     # output[1] = 0
#     # print(output)

#     # visJoints = torch.ones((2, 2))
#     # visJoints[1] = 0
#     # print(visJoints)
#     # loss = JointsMSELoss()
#     # calcLoss = loss(target, output, visJoints)
#     # print(calcLoss.item())

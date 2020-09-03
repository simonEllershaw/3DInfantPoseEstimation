import torch.nn as nn
import torch


class JointsMSELoss(nn.Module):
    """
        Calculates MSE on 2 heatmaps, ignoring non visible joints
        Adapted from: https://github.com/microsoft/human-pose-estimation.pytorch
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

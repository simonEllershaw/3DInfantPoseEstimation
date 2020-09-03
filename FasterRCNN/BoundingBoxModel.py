from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
import os
import sys
import matplotlib.pyplot as plt
from torchvision import transforms

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import DataSets.Utils.Visualisation as vis


class BoundingBoxModel(object):
    """
        Container for Pytorch Faster R-CNN (https://pytorch.org/docs/stable/torchvision/models.html)
        with added utility functions
    """
    def __init__(self, device, modelToLoadFname=None):
        self.device = device
        self.model = BoundingBoxModel.loadFasterRCNNModel(device)
        if modelToLoadFname:
            checkpoint = torch.load(modelToLoadFname)
            self.model.load_state_dict(checkpoint["model_state_dict"])

    @staticmethod
    def loadFasterRCNNModel(device):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        return model.to(device)

    def getCentreAndScale(self, image):
        imageTensor = transforms.ToTensor()(image).to(self.device)
        bbox = self.outputBoundingBox(imageTensor)[0]
        # From bounding box extract centre and scale metrics
        centre = (bbox[2:] + bbox[:2]) // 2
        bboxDimensions = bbox[2:] - bbox[:2]
        scale = torch.max(bboxDimensions).item() / 200  # scale in relation to 200px
        return scale, centre.cpu().numpy()

    def outputBoundingBox(self, images):
        output = self.model(images)

        # Reform output into size [N,4]
        tensorOutputs = torch.zeros(
            (len(output), 4),
            device=self.device,
            requires_grad=False,
            dtype=torch.float32,
        )
        # Take highest scoring box (index 0)
        # If no prediction is made proposed box = [0,0,0,0]
        for i in range(len(output)):
            proposedBoxes = output[i]["boxes"]
            if proposedBoxes.size()[0] != 0:
                tensorOutputs[i] = proposedBoxes[0].round()
            else:
                # If no prediction made bounding box = image size
                tensorOutputs[i][2] = images[i].size()[1]
                tensorOutputs[i][3] = images[i].size()[0]
                print("No prediction made")
        return tensorOutputs

    def calcLosses(self, inputs, targets):
        return self.model(inputs, targets)

    def visualiseAnOutput(self, sample, fname):
        self.model.eval()
        image, target, meta = sample
        output = self.outputBoundingBox([sample[0]], self.device)[0]

        plt.figure()
        ax = plt.subplot(1, 2, 1)
        meta["videoPath"] = "Ground Truth"
        meta["frameNumber"] = ""
        vis.plotBbox(ax, target["boxes"][0])

        ax = plt.subplot(1, 2, 2)
        meta["videoPath"] = "Predicted"
        vis.plotBbox(ax, output)
        plt.savefig(fname)

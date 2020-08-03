from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from torch.optim import Adam
import numpy as np
from torchvision import transforms
import cv2

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.hesseSynthetic import HesseSyntheticDataset
import DataSets.transforms
import trainer
from DataSets.TargetType import TargetType


class BoundingBoxModel(object):
    def __init__(self, device, modelToLoadFname=None):
        self.device = device
        self.model = BoundingBoxModel.loadFasterRCNNModel(device)
        if modelToLoadFname:
            checkpoint = torch.load(modelToLoadFname)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.imageFileName = "../images/FasterRCNNHesse.png"

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
            (len(output), 4), device=self.device, requires_grad=False, dtype=torch.float32
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

    def visualiseAnOutput(self, sample):
        self.model.eval()
        image, target, meta = sample
        output = self.outputBoundingBox([sample[0]], self.device)[0]

        plt.figure()
        ax = plt.subplot(1, 2, 1)
        meta["videoPath"] = "Ground Truth"
        meta["frameNumber"] = ""
        HesseSyntheticDataset.visualiseSample(sample, ax)

        ax = plt.subplot(1, 2, 2)
        meta["videoPath"] = "Predicted"
        HesseSyntheticDataset.visualiseSample((image, {"boxes": [output]}, meta), ax)

        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        plt.savefig(os.path.join(__location__, self.imageFileName))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batchSize = 4
    dataLoaders = dataloaders.getMPIInfDataLoader(batchSize, 256, TargetType.bbox)
    # dataLoaders = dataloaders.getHesseDataLoader(batchSize, 256)

    bboxModel = BoundingBoxModel(device)
    params = [p for p in bboxModel.model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=1e-4)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=30,
    #                                                gamma=0.1)

    # Create Directory
    dateAndTime = datetime.now().strftime("%d_%m_%H_%M")
    directory = os.path.join(os.path.dirname(__file__), f"savedModels/{dateAndTime}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    trainer.train_model(bboxModel, dataLoaders, device, optimizer, directory)

    # bboxModel = BoundingBoxModel(
    #     device, "/homes/sje116/Diss/FasterRCNN/savedModels/Hesse/model.tar",
    # )
    # bboxModel.model.eval()
    # # data = dataloaders.getHesseDataLoader(4, 256, bboxModel)
    # # for batch in data["train"]:
    # #     print("Batch")
    # dataSet = HesseSyntheticDataset("train", 256, bboxModel)

    # sample = dataSet[0]

    # plt.figure()
    # ax = plt.subplot(1, 1, 1)
    # HesseSyntheticDataset.visualiseSample(sample, ax, boundingBox=False)

    # __location__ = os.path.realpath(
    #     os.path.join(os.getcwd(), os.path.dirname(__file__))
    # )
    # plt.savefig(os.path.join(__location__, "../images/HesseCropped.png"))


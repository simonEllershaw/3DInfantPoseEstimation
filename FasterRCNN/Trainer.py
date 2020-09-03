import time
import torch
import os
from torchvision.ops.boxes import box_iou
import sys
from torch.optim import Adam
from datetime import datetime

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.Concrete.MINI_RGBDDataset import MINI_RGBDDataset
from DataSets.Utils.TargetType import TargetType
from FasterRCNN.BoundingBoxModel import BoundingBoxModel
import FasterRCNN.Trainer as Trainer

"""
    Training function for Faster R-CNN models
"""


def train_model(bboxModel, dataloaders, device, optimizer, directory):

    dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}
    best_IOU = 0
    numberEpochsWtNoImprovement = 0
    epoch = 0

    while numberEpochsWtNoImprovement < 3:
        trainLoss = 0
        valLoss = 0
        epochStart = time.time()

        for phase in ["train", "val"]:
            if phase == "train":
                bboxModel.model.train()
            else:
                bboxModel.model.eval()
            running_loss = 0.0
            for inputs, targets, meta in dataloaders[phase]:
                inputs = list(img.to(device) for img in inputs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    if phase == "train":
                        # Loss defined by Faster R-CNN model
                        loss_dict = bboxModel.calcLosses(inputs, targets)
                        loss = sum(loss for loss in loss_dict.values())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        # Use IoU as intuative accuracy metric
                        output = bboxModel.outputBoundingBox(inputs)
                        # Reform targets to size and format [N,4]
                        tensorTargets = torch.stack([o["boxes"][0] for o in targets])
                        loss = torch.mean(box_iou(output, tensorTargets))
                    running_loss += loss.item()

            # Record losses and update saved model accordingly
            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == "train":
                trainLoss = epoch_loss
            elif phase == "val":
                valLoss = epoch_loss
                # Improvement has to be at least by 0.1% to save model
                if epoch_loss > best_IOU * 1.001:
                    best_IOU = epoch_loss
                    saveCheckpoint(
                        epoch, bboxModel.model, optimizer, best_IOU, directory
                    )
                    numberEpochsWtNoImprovement = 0
                else:
                    numberEpochsWtNoImprovement += 1

        epochEnd = time.time()
        epochTime = epochEnd - epochStart

        # Write stats to file at end of each epoch
        with open(os.path.join(directory, "metrics.txt"), "a+") as outfile:
            outfile.write(f"{epoch} {trainLoss} {valLoss} {best_IOU} {epochTime} \n")
        epoch += 1


def saveCheckpoint(epoch, model, optimizer, loss, directory):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(directory, "model.tar"),
    )


if __name__ == "__main__":
    # Setup for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batchSize = 1

    dataLoaders = MINI_RGBDDataset.getDataLoader(batchSize, TargetType.bbox)

    bboxModel = BoundingBoxModel(device)
    params = [p for p in bboxModel.model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=1e-4)

    # Create Directory
    dateAndTime = datetime.now().strftime("%d_%m_%H_%M")
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    directory = os.path.join(__location__, f"../../SavedModels/{dateAndTime}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    Trainer.train_model(bboxModel, dataLoaders, device, optimizer, directory)

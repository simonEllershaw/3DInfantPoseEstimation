import time
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torchvision.ops.boxes import box_iou

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from DataSets.hesseSynthetic import HesseSyntheticDataset
from DataSets.MPI_INF_3DHP import MPI_INF_3DHP
from DataSets.TargetTypes import TargetTypes

def train_model(bboxModel, dataloaders, device, optimizer, directory):

    dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}
    best_IOU = 0
    # numberEpochsWtNoImprovement = 0
    # epoch = 0

    for epoch in range(10):  # numberEpochsWtNoImprovement < 2:
        trainLoss = 0
        valLoss = 0
        epochStart = time.time()
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                bboxModel.model.train()
            else:
                bboxModel.model.eval()
            running_loss = 0.0
            # Iterate over data.
            for inputs, targets, meta in dataloaders[phase]:
                inputs = list(img.to(device) for img in inputs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    if phase == "train":
                        loss_dict = bboxModel.calcLosses(inputs, targets)
                        loss = sum(loss for loss in loss_dict.values())
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        output = bboxModel.outputBoundingBox(inputs)
                        # Reform targets to size and format [N,4]
                        tensorTargets = torch.stack([o["boxes"][0] for o in targets])
                        loss = torch.mean(box_iou(output, tensorTargets))

                    running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == "train":
                trainLoss = epoch_loss
            elif phase == "val":
                valLoss = epoch_loss
                # Improvement has to be at least by 0.1%
                if epoch_loss > best_IOU * 1.001:
                    best_IOU = epoch_loss
                    saveCheckpoint(epoch, bboxModel.model, optimizer, best_IOU, directory)
                    # numberEpochsWtNoImprovement = 0
                # else:
                #     numberEpochsWtNoImprovement += 1

        epochEnd = time.time()
        epochTime = epochEnd - epochStart

        # Write stats to file at end of each epoch
        with open(os.path.join(directory, "metrics.txt"), "a+") as outfile:
            outfile.write(f"{epoch} {trainLoss} {valLoss} {best_IOU} {epochTime} \n")

        #  epoch += 1


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
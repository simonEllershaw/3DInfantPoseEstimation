"""
    Training functionality for pose estimation models
"""

import time
import torch
import os
import sys
from datetime import datetime
from torch.optim import Adam

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

import PoseEstimation.ModelArchs.ModelGenerator as ModelGenerator
import PoseEstimation.Core.Inference as Inference


def train_model(
    model,
    dataloaders,
    device,
    criterion,
    optimizer,
    directory,
    lrScheduler=None,
    pose2DModel=None,
):
    dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}

    best_loss = float("inf")
    numberEpochsWtNoImprovement = 0
    epoch = 0
    # for epoch in range(num_epochs):
    while numberEpochsWtNoImprovement < 3:
        epoch += 1
        trainLoss = 0
        valLoss = 0
        epochStart = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for source, targets, meta in dataloaders[phase]:
                source = source.to(device)
                targets = targets.to(device)
                visJoints = meta["visJoints"].to(device)

                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # If 2 models provided end to end training occurs
                    if pose2DModel is not None:
                        outputs = pose2DModel(source)
                        preds = outputs.detach().cpu().numpy()
                        predCoords = Inference.postProcessPredictions(
                            preds, meta["centre"].numpy(), meta["scale"].numpy(), 64
                        )
                        source = torch.tensor(predCoords).to(device).view(-1, 32)
                    outputs = model(source)
                    loss = criterion(outputs, targets, visJoints)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == "train":
                trainLoss = epoch_loss
            elif phase == "val":
                valLoss = epoch_loss
                # Improvement has to be at least by 0.1%
                if epoch_loss < best_loss * 0.999:
                    best_loss = epoch_loss
                    saveCheckpoint(epoch, model, optimizer, loss, directory)
                    numberEpochsWtNoImprovement = 0
                else:
                    numberEpochsWtNoImprovement += 1

        epochEnd = time.time()
        epochTime = epochEnd - epochStart
        if lrScheduler:
            lrScheduler.step()

        # Write stats to file at end of each epoch
        with open(os.path.join(directory, "metrics.txt"), "a+") as outfile:
            outfile.write(f"{epoch} {trainLoss} {valLoss} {best_loss} {epochTime} \n")


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
    # Setup to finetune 2D pose model on MINI_RGBD dataset
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batchSize = 16
    dataLoaders, model, criterion = ModelGenerator.getHesse2DPoseTrainingObjects(
        batchSize, device
    )

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    checkPointFname = os.path.join(__location__, "../../SavedModels/MPII2D/model.tar")
    checkpoint = torch.load(checkPointFname)
    print("load")
    model.load_state_dict(checkpoint["model_state_dict"])

    paramsToUpdate = model.parameters()
    optimizer = Adam(paramsToUpdate, lr=1e-4)
    # lrScheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer=optimizer, gamma=0.96
    # )

    # Setup Directory
    dateAndTime = datetime.now().strftime("%d_%m_%H_%M")
    print(dateAndTime)
    directory = os.path.join(__location__, f"../../SavedModels/{dateAndTime}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    train_model(
        model,
        dataLoaders,
        device,
        criterion,
        optimizer,
        directory
        #  lrScheduler, pose2DModel
    )

import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)


# From https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(
    model,
    dataloaders,
    device,
    criterion,
    optimizer,
    directory,
    num_epochs=25,
):
    dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}
    print(dataset_sizes)
    best_loss = float("inf")
    numberEpochsWtNoImprovement = 0
    epoch = 0
    # for epoch in range(num_epochs):
    while(True):
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
            t2 = time.time()
            # Iterate over data.
            for images, targets, meta in dataloaders[phase]:
                # t1 = time.time()
                # print(f"Load time {t1 - t2}")
                images = images.to(device)
                targets = targets.to(device)
                visJoints = meta["visJoints"].to(device)

                # zero the parameter gradients

                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
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

        # Write stats to file at end of each epoch
        with open(os.path.join(directory, "metrics.txt"), "a+") as outfile:
            outfile.write(f"{epoch} {trainLoss} {valLoss} {best_loss} {epochTime} \n")
        
        # Early exit if no improvement for 2 epochs on val set
        if numberEpochsWtNoImprovement > 2:
            print("Exiting")
            break


def saveCheckpoint(epoch, model, optimizer, loss, directory):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        },
        os.path.join(directory, "model.tar"),
    )

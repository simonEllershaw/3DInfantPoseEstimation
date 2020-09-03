import torch
import numpy as np
import matplotlib.patches as patches

"""
Module for all visualisations of Bboxes, 3D, 2D, Heatmaps and Images
"""


def plotBbox(ax, bbox):
    x = bbox[0]
    y = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    rect = patches.Rectangle((x, y), width, height, edgecolor="b", fill=False)
    ax.add_patch(rect)


def plot3DJoints(ax, joints3D, connectedJoints, jointColours):

    if torch.is_tensor(joints3D):
        joints3D = joints3D.detach().cpu().numpy()
    joints3D = joints3D.reshape(-1, 3)

    # Plot 3D skeleon
    for i in np.arange(len(connectedJoints)):
        x, y, z = [
            np.array(
                [
                    joints3D[connectedJoints[i, 0], j],
                    joints3D[connectedJoints[i, 1], j],
                ]
            )
            for j in range(3)
        ]
        ax.plot(x, y, z, lw=2, c=jointColours[i])

    # Plot coordiantes
    ax.scatter(joints3D[:, 0], joints3D[:, 1], joints3D[:, 2], c="black")

    # Camera properties
    ax.set_xlim3d([-250, 250])
    ax.set_ylim3d([-250, 250])
    ax.set_zlim3d([-250, 250])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    elev = -120
    azim = -127.5
    ax.view_init(elev, azim)


def plotHeatmap(ax, heatmaps):
    # Plot target 2D joint gaussians
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.cpu().numpy()
    levels = np.arange(0.4, 1.2, 0.2, dtype="float32")
    upsampledJoints = heatmaps.repeat(4, axis=1).repeat(4, axis=2)
    for joint in upsampledJoints:
        ax.contourf(joint, levels)


def plotImage(ax, image):
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(image)


def plot2DJoints(ax, joints2D, connectedJoints, jointColours, visJoints=None):
    # Plot skeleton
    for i in np.arange(len(connectedJoints)):
        joint1 = connectedJoints[i, 0]
        joint2 = connectedJoints[i, 1]
        if visJoints is None or (visJoints[joint1] == 1 and visJoints[joint2] == 1):
            x, y = [
                np.array(
                    [
                        joints2D[connectedJoints[i, 0], j],
                        joints2D[connectedJoints[i, 1], j],
                    ]
                )
                for j in range(2)
            ]
            ax.plot(x, y, lw=2, c=jointColours[i])

    # Plot joint coordiantes
    for i in range(len(joints2D)):
        scatterColour = "black" if visJoints is None or visJoints[i] == 1 else "orange"
        ax.scatter(joints2D[i, 0], joints2D[i, 1], c=scatterColour)
        # ax.text(
        #     joints2D[i, 0], joints2D[i, 1]-5, i)

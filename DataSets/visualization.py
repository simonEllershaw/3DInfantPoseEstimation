import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

def plot3DJoints(ax, joints3D, connectedJoints, jointColours):
    plt.tight_layout()

    if torch.is_tensor(joints3D):
        joints3D = joints3D.detach().cpu().numpy()
    joints3D = joints3D.reshape(-1, 3)

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

    ax.scatter(
        joints3D[:, 0], joints3D[:, 1], joints3D[:, 2], c="black")
    
    # for i in range(len(joints3D)):
    #     ax.text(joints3D[i, 0], joints3D[i, 1], joints3D[i, 2], f"{i}")

    # RADIUS = 0.25  # space around the subject
    # xroot, yroot = joints3D[0, 0], joints3D[0, 1]
    # zmax = np.amax(joints3D, axis=0)[2]
    # ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    # ax.set_zlim3d([-RADIUS + 2 * zmax, zmax])
    # ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    # ax.set_xlim3d([-1000, 1000])
    # ax.set_ylim3d([-1000, 1000])
    # ax.set_xlim3d([-1000, 1000])
    ax.set_xlim3d([-250, 250])
    ax.set_ylim3d([-250, 250])
    ax.set_zlim3d([-250, 250])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    elev = -120
    azim = -127.5
    ax.view_init(elev, azim)

    # ax.axis("off")
    # if torch.is_tensor(image):
    #     image = image.permute(1, 2, 0).cpu().numpy()

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

def plot2DJoints(ax, joints2D, connectedJoints, jointColours):
    for i in np.arange(len(connectedJoints)):
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

    ax.scatter(
        joints2D[:, 0], joints2D[:, 1], c="black")
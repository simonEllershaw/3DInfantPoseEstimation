import matplotlib.pyplot as plt
import torch
import numpy as np

def plot3DJoints(ax, joints3D, connectedJoints, title=None):
    plt.tight_layout()
    if title:
        ax.set_title(title)
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
        ax.plot(x, y, z, lw=2, c="red")

    ax.scatter(
        joints3D[:, 0], joints3D[:, 1], joints3D[:, 2], c="black")

    # RADIUS = 0.25  # space around the subject
    # xroot, yroot = joints3D[0, 0], joints3D[0, 1]
    # zmax = np.amax(joints3D, axis=0)[2]
    # ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    # ax.set_zlim3d([-RADIUS + 2 * zmax, zmax])
    # ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-1000, 1000])
    ax.set_xlim3d([-1000, 1000])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    elev = -70
    azim = -90
    ax.view_init(elev, azim)

    # ax.axis("off")
    # if torch.is_tensor(image):
    #     image = image.permute(1, 2, 0).cpu().numpy()
    # ax.imshow(image)
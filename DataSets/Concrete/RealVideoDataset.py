import os
from PIL import Image
import torch
import copy
import numpy as np
import re
import cv2
import sys
from torchvision import transforms
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

import DataSets.Utils.Transforms
import DataSets.Utils.Visualisation as vis
from DataSets.Abstract.JointsDataset import JointsDataset
import DataSets.Utils.Config as cfg


class RealVideoDataset(JointsDataset):
    """
        Class made quickly to load real MAVHEA video frames
        Does not follow full hierarchy and as such has code repetition that could be removed
    """

    def __init__(self):
        JointsDataset.__init__(self, "test", cfg.MPII["numJoints"])
        self.videoDirectory = cfg.MAHVEA["videoDirectory"]
        self.db = self._get_db()

    def _get_db(self):
        db = []
        # Iterate through imageFiles sorted in ascending order
        for imageFile in sorted(
            os.listdir(self.videoDirectory), key=RealVideoDataset.getImageFrameNumber
        ):
            print(imageFile)
            imagePath = os.path.join(self.videoDirectory, imageFile)

            db.append({"imagePath": imagePath})
        return db

    # https://stackoverflow.com/questions/17336943/removing-non-numeric-characters-from-a-string
    def getImageFrameNumber(string):
        return int(re.sub("[^0-9]", "", string))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        anno = copy.deepcopy(self.db[idx])

        # Image cropped and resized to be equal to MINI-RGBD input
        # This is currently hard coded
        image = Image.open(anno["imagePath"])
        height = 1280
        width = 960
        image = image.crop((120, 640, 120 + width, 640 + height))
        image = image.resize((width // 2, height // 2))
        image = np.array(image)

        # Centre and scale are just cropped image dimensions
        centre = np.array(image.shape[:-1]) // 2
        scale = np.max(image.shape[:-1]) / 200

        processedImage = self.preProcessImage(image, [centre[1], centre[0]], scale)

        meta = {
            "imagePath": anno["imagePath"],
            "centre": centre,
            "scale": scale,
            "cropImage": image,
        }

        # No target as no 3D ground truths are available
        return processedImage, "None", meta

    def preProcessImage(self, image, center, scale):
        # Basic pre-processing of image
        rotation = 0
        trans = DataSets.Utils.Transforms.get_affine_transform(
            center, scale, rotation, self.imageSize
        )
        image = cv2.warpAffine(
            image, trans, (self.imageSize, self.imageSize), flags=cv2.INTER_LINEAR
        )
        image = transforms.ToTensor()(image)

        return image

    def getDataLoader(batchSize):
        # Only one data type so no dictionary
        # Shuffle = False so outputs generated in order
        num_workers = 4
        dataloaders = torch.utils.data.DataLoader(
            RealVideoDataset(),
            batch_size=batchSize,
            shuffle=False,
            num_workers=num_workers,
        )
        return dataloaders


if __name__ == "__main__":
    # Test function to load and visualise a sample
    data = RealVideoDataset()
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    fig, ax = plt.subplots()
    vis.plotImage(ax, data[0][2]["cropImage"])
    plt.savefig(os.path.join(__location__, "../../Images/video.png"))

import numpy as np

generic = {"imageSize": 256, "heatMapSize": 64}

MPI_INF = {
    "pelvicIndex": 4,
    "numJoints": 28,
    "basePath": "/vol/bitbucket/sje116/mpi-inf-3dhp/mpi_inf_3dhp/",
    "modeSubjects": {
        "train": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "val": ["S7", "S8"],
    },
    "cameras": [1, 2, 3, 4, 5, 6, 7, 8],
    "sequences": ["Seq1", "Seq2"],
    "numFrames": {
        "S1": {"Seq1": 6416, "Seq2": 12430},
        "S2": {"Seq1": 6502, "Seq2": 6081},
        "S3": {"Seq1": 12488, "Seq2": 12283},
        "S4": {"Seq1": 6171, "Seq2": 6675},
        "S5": {"Seq1": 12820, "Seq2": 12312},
        "S6": {"Seq1": 6188, "Seq2": 6145},
        "S7": {"Seq1": 6239, "Seq2": 6320},
        "S8": {"Seq1": 6468, "Seq2": 6054},
    },
    "hasPersonSegmentation": {
        "S1": {"Seq1": False, "Seq2": False},
        "S2": {"Seq1": False, "Seq2": True},
        "S3": {"Seq1": False, "Seq2": True},
        "S4": {"Seq1": False, "Seq2": False},
        "S5": {"Seq1": False, "Seq2": True},
        "S6": {"Seq1": False, "Seq2": True},
        "S7": {"Seq1": True, "Seq2": False},
        "S8": {"Seq1": True, "Seq2": False},
    },
    "connectedJoints": np.array(
        [
            # Spine
            [4, 3],
            [2, 3],
            [0, 2],
            [0, 1],
            [1, 5],
            [5, 6],
            [6, 7],
            # Left leg
            [4, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            [21, 22],
            # Left Arm
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            # Right leg
            [27, 26],
            [26, 25],
            [25, 24],
            [24, 23],
            [23, 4],
            # Right Arm
            [1, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [16, 17],
        ]
    ),
    "jointColours": [
        "green",
        "green",
        "green",
        "green",
        "green",
        "green",
        "green",
        "red",
        "red",
        "red",
        "red",
        "red",
        "red",
        "red",
        "red",
        "red",
        "red",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
    ],
    "MPIIMapping": [25, 24, 23, 18, 19, 20, 4, 1, 5, 6, 16, 15, 14, 9, 10, 11],
    "hesseMappings": [
        4,
        18,
        23,
        3,
        19,
        24,
        2,
        20,
        25,
        0,
        22,
        27,
        5,
        8,
        13,
        6,
        9,
        14,
        10,
        15,
        11,
        16,
        12,
        17,
        7,
    ],
    "jointsNotInHesse": (1, 21, 26),
    "neckIndex": 6,
    "headIndex": 7,
    "PCKhThreshold": 150,
}

MINI_RGBD = {
    "basePath": "/vol/bitbucket/sje116/Hesse/",
    "numJoints": 25,
    "modeDatasets": {
        "train": np.arange(1, 9),
        "val": np.arange(9, 11),
        "test": np.arange(11, 13),
    },
    "connectedJoints": np.array(
        [
            # Spine
            [0, 3],
            [3, 6],
            [6, 9],
            [9, 12],
            [12, 15],
            [15, 24],
            # Left Leg
            [0, 1],
            [1, 4],
            [4, 7],
            [7, 10],
            # Left arm
            [12, 13],
            [13, 16],
            [16, 18],
            [18, 20],
            [20, 22],
            # Right leg
            [0, 2],
            [2, 5],
            [5, 8],
            [8, 11],
            # Right arm
            [12, 14],
            [14, 17],
            [17, 19],
            [19, 21],
            [21, 23],
        ]
    ),
    "jointColours": [
        "green",
        "green",
        "green",
        "green",
        "green",
        "green",
        "red",
        "red",
        "red",
        "red",
        "red",
        "red",
        "red",
        "red",
        "red",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
        "blue",
    ],
    "cameraIntrinsics": np.array(
        (
            [588.67905803875317, 0, 322.22048191353628],
            [0, 590.25690113005601, 237.46785983766890],
            [0, 0, 1],
        )
    ),
    "neckIndex": 12,
    "headIndex": 15,
    "pelvicIndex": 0,
    "numFramesPerSequence": 1000,
    "videoPCKhThresholds": [
        None,  # Videos start at 1
        {"2D": 11.21, "3D": 24.9},
        {"2D": 11.39, "3D": 26.3},
        {"2D": 11.96, "3D": 25.3},
        {"2D": 11.13, "3D": 27.8},
        {"2D": 7.75, "3D": 20.6},
        {"2D": 9.10, "3D": 23.4},
        {"2D": 12.77, "3D": 28.2},
        {"2D": 13.34, "3D": 26.2},
        {"2D": 12.75, "3D": 29.0},
        {"2D": 11.23, "3D": 26.1},
        {"2D": 12.17, "3D": 28.6},
        {"2D": 13.93, "3D": 30.1},
    ],
    "jointNames": [
        "pelvis",
        "leftThigh",
        "rightThigh",
        "spine",
        "leftCalf",
        "rightCalf",
        "spine1",
        "leftFoot",
        "rightFoot",
        "spine2",
        "leftToes",
        "rightToes",
        "neck",
        "leftShoulder",
        "rightShoulder",
        "head",
        "leftUpperArm",
        "rightUpperArm",
        "leftForeArm",
        "rightForeArm",
        "leftHand",
        "rightHand",
        "leftFingers",
        "rightFingers",
        "noseVertex",
    ],
    "MPIIMapping": [8, 5, 2, 1, 4, 7, 0, 9, 12, 24, 21, 19, 17, 16, 18, 20],
}

MPII = {
    "numJoints": 16,
    "modeDatasets": {"train": ["train", "trainval"], "val": ["valid"]},
    "annotationFileDirectory": "/vol/bitbucket/sje116/MPII/annotations",
    "imageDirectory": "/vol/bitbucket/sje116/MPII/images/",
    "connectedJoints": np.array(
        [
            [0, 1],
            [1, 2],
            [2, 6],
            [3, 4],
            [4, 5],
            [3, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [8, 12],
            [12, 11],
            [11, 10],
            [8, 13],
            [13, 14],
            [14, 15],
        ]
    ),
    "neckIndex": 8,
    "headIndex": 9,
    "pelvicIndex": 6,
    "jointNames": [
        "r ankle",
        "r knee",
        "r hip",
        "l hip",
        "l knee",
        "l ankle",
        "pelvis",
        "thorax",
        "upper neck",
        "head top",
        "r wrist",
        "r elbow",
        "r shoulder",
        "l shoulder",
        "l elbow",
        "l wrist",
    ],
    "jointColours": [
        "blue",
        "blue",
        "blue",
        "red",
        "red",
        "red",
        "green",
        "green",
        "green",
        "blue",
        "blue",
        "blue",
        "red",
        "red",
        "red",
    ],
}
MAHVEA = {
    "labelsFname": "/vol/bitbucket/sje116/InfantData/label.json",
    "baseDirectory": "/vol/bitbucket/sje116/InfantData",
    "MPIIMapping": [6, 7, 8, 3, 4, 5, 0, 0, 0, 0, 11, 10, 9, 2, 1, 0],
    "nonMappedJoints": [6, 7, 8, 9],
    "numJoints": 12,
    "videoDirectory": "/vol/bitbucket/sje116/video",
}
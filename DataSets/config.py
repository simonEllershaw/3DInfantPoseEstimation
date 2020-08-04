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
            [27, 26],
            [26, 25],
            [25, 24],
            [24, 23],
            [23, 4],
            [4, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            [21, 22],
            [4, 3],
            [2, 3],
            [0, 2],
            [0, 1],
            [1, 8],
            [1, 13],
            [1, 5],
            [5, 6],
            [6, 7],
            [13, 14],
            [14, 15],
            [15, 16],
            [16, 17],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
        ]
    ),
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
    "annotionsFname": "/homes/sje116/Diss/DataSets/MPI-INF-3DHPUtils/annotations",
}

Hesse = {
    "basePath": "/vol/bitbucket/sje116/Hesse/",
    "numJoints": 25,
    "modeDatasets": {
        "train": np.arange(1, 9),
        "val": np.arange(9, 11),
        "test": np.arange(11, 13),
    },
    "connectedJoints": np.array(
        [
            [1, 2],
            [2, 5],
            [5, 8],
            [8, 11],
            [10, 14],
            [14, 17],
            [17, 19],
            [19, 21],
            [21, 23],
            [1, 3],
            [3, 6],
            [6, 9],
            [9, 12],
            [10, 15],
            [15, 18],
            [18, 20],
            [20, 22],
            [22, 24],
            [1, 4],
            [4, 7],
            [7, 10],
            [10, 13],
            [13, 16],
            [16, 25],
        ]
    )
    - 1,
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
}

MPII = {
    "numJoints": 16,
    "modeDatasets": {"train": ["train", "trainval"], "val": ["valid"]},
    "annotationFileDirectory": "/homes/sje116/Diss/DataSets/MPIIAnnotations",
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
}

import enum

"""
    Enum defintion of 4 target types
"""


class TargetType(enum.Enum):
    joint2D = 1
    joint3D = 2
    bbox = 3
    endToEnd = 4

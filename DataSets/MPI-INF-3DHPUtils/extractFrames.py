# https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/

# and Extract Frames
import cv2
import os

# Function to extract frames
def FrameCapture(videoPath, outputDirectory):

    # Path to video file
    vidObj = cv2.VideoCapture(videoPath)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while True:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if not success:
            break
        # Saves the frames with frame-count
        outputFname = os.path.join(outputDirectory, f"frame{count:05}.jpg")
        cv2.imwrite(outputFname, image)
        count += 1


if __name__ == "__main__":
    baseFname = "/vol/bitbucket/sje116/mpi-inf-3dhp/mpi_inf_3dhp"
    subjects = ["S1"]#["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    sequences = ["Seq2"]#["Seq1", "Seq2"]
    videoFiles = ["FGmasks", "imageSequence"]
    videos = [0, 1, 2, 4, 5, 6, 7, 8]
    for subject in subjects:
        for sequence in sequences:
            for videoFile in videoFiles:
                directory = os.path.join(baseFname, subject, sequence, videoFile)
                for video in videos:
                    videoPath = os.path.join(directory, f"video_{video}.avi")
                    print(videoPath)
                    outputDirectory = os.path.join(directory, f"{video:02}")
                    if not os.path.exists(outputDirectory):
                        os.mkdir(outputDirectory)
                    FrameCapture(videoPath, outputDirectory)
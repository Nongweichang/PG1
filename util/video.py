from time import sleep

import cv2


class video:
    def __init__(self, videoPath, dirPath):
        self.dirPath = dirPath
        self.videoPath = videoPath

    def getFrames(self, interval=0):
        videoCapture = cv2.VideoCapture(self.videoPath)
        i = 0
        while True:
            success, frame = videoCapture.read()
            if not success:
                print("end")
                break
            else:
                if i % interval == 0:
                    cv2.imwrite(self.dirPath + "/" + str(i) + ".jpg", frame)
            i += 1

    @staticmethod
    def camera():
        cam = cv2.VideoCapture(0)
        while True:
            success, frame = cam.read()
            if success:
                return frame

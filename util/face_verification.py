import os
import time

import cv2

from util.Constant import Constant
from util.LoadData import load_img, getFace, getEye, resizeImg
from util.Model import Model
from util.myException import MyException
from util.video import video


class face_verification:
    def __init__(self):
        if not os.path.exists(Constant.model):
            raise MyException("无模型文件")
        self.model = Model()
        self.model.load()
        img, count, EYES = load_img(Constant.baseImg)
        if count == 0:
            raise MyException("无基准照片")
        self.base = self.model.predict(img[0])
        # print(len(self.model.predict(img[0])))

    def analysis(self, unknown):
        count = [0, 0, 0, 0, 0]
        x = []
        for i in range(len(unknown)):
            x = (self.base[i] - unknown[i]) * (self.base[i] - unknown[i])
        for i in range(len(x)):
            if x[i] <= 1e-40:
                count[0] += 1
            elif x[i] <= 1e-30:
                count[1] += 1
            elif x[i] <= 1e-20:
                count[2] += 1
            elif x[i] <= 1e-10:
                count[3] += 1
            elif x[i] > 1e-10:
                count[4] += 1
        return count

    def Picture_recognition_mode(self):
        ans = 0.0
        IMGs, count, EYEs = load_img(Constant.unknownIMGs)
        for i in range(len(IMGs)):
            img = IMGs[i]
            t = self.model.predict(img)
            aresult = self.analysis(t)
            print(aresult)
            if aresult[0] >= 48:
                ans += 1
                if len(EYEs[i]) == 0:
                    print("no eye")
                    ans -= 0.3
        return ans / count

    def Real_time_recognition_mode(self, Duration, interval=600):
        os.system("cls")
        print("10秒后开始检测")
        time.sleep(10)
        print("开始")
        start = time.time()
        count = 0
        ans = 0.0
        while True:
            img = video.camera()
            count += 1
            img = getFace(img)
            eye = getEye(img)
            img = resizeImg(img)
            for face in img:
                t = self.model.predict(face)
                aresult = self.analysis(t)
                if aresult[0] >= 48:
                    ans += 1
                    if len(eye) == 0:
                        ans -= 0.3
                    break
            end = time.time()
            if end - start >= Duration * 60:
                break
            time.sleep(interval)
        return ans / count

    def Video_mode(self, path):
        v = video(path, Constant.unknownIMGs)
        v.getFrames(30)
        return self.Picture_recognition_mode()

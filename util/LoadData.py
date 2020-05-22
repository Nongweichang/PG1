# -*- coding: utf-8 -*-


import os
import cv2

from util.Constant import Constant

face_cascade = cv2.CascadeClassifier("res/xml/haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("res/xml/haarcascade_eye_tree_eyeglasses.xml")


# 用CV2 目录内生成的 不能用github上的


def resizeImg(img, width=Constant.imageSize, height=Constant.imageSize):
    top, bottom, left, right = (0, 0, 0, 0)  # 储存黑边的宽度,默认为0

    h, w, _ = img.shape  # 获取原图的长宽
    longer_edge = max(h, w)  # 获取较长一边的值

    # 计算黑边的值
    if h < longer_edge:
        dh = longer_edge - h
        top = dh // 2
        bottom = dh - top

    elif w < longer_edge:

        dw = longer_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # 填充黑边
    constant = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 返回调整后的图片
    return cv2.resize(constant, (height, width))


def getEye(img):
    t = eye_cascade.detectMultiScale(img)
    return t


def getFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(5, 5), )

    if len(faces) is not 0:
        x = faces[0][0]
        y = faces[0][1]
        width = faces[0][2]
        height = faces[0][3]
        return img[y:y + height, x:x + width]
    return None


IMG = []  # 图片列表
EYE = []


def load_img(path):
    IMG.clear()
    EYE.clear()
    i = 0
    for dir_item in os.listdir(path):
        # print('loading')
        full_path = os.path.abspath(os.path.join(path, dir_item))
        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            load_img(full_path)
        else:
            if dir_item.endswith("png") or dir_item.endswith(".jpg") or dir_item.endswith(".bmp"):
                i += 1
                # print(dir_item)
                image = cv2.imread(full_path)
                image = getFace(image)
                if image is not None:
                    EYE.append(getEye(image))
                    image = resizeImg(image)
                    IMG.append(image)
                    # print(path)
    return IMG, i, EYE

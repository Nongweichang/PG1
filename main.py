import os

import cv2

from util.LoadData import getFace, getEye
from util.face_verification import face_verification
from util.myException import MyException

try:
    fv = face_verification()
    ans = fv.Picture_recognition_mode()
    print(ans)
except MyException as obj:
    print(obj.msg)

# # os.system("pause")
# img = cv2.imread("123.jpg")
# img = getFace(img)
# t=getEye(img)
# print(len(t))
# eye_cascade = face_cascade = cv2.CascadeClassifier("res/xml/haarcascade_eye_tree_eyeglasses.xml")
# eye = eye_cascade.detectMultiScale(img)
# print(eye)

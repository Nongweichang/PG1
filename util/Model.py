from keras.engine.saving import load_model
from keras import backend as K

from util.Constant import Constant
from util.LoadData import resizeImg


class Model:
    def __init__(self):
        self.model = None

    def load(self, path=Constant.model):
        self.model = load_model(path)

    def predict(self, img):
        if K.image_data_format() == 'channels_first' and img.shape != (1, 3, Constant.imageSize, Constant.imageSize):
            img = resizeImg(img)
            img = img.reshape((1, 3, Constant.imageSize, Constant.imageSize))
        elif K.image_data_format() == 'channels_last' and img.shape != (1, Constant.imageSize, Constant.imageSize, 3):
            img = resizeImg(img)
            img = img.reshape((1, Constant.imageSize, Constant.imageSize, 3))
        img = img.astype('float32')
        img /= 255
        result = self.model.predict_proba(img)
        return result

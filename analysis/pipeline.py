import tensorflow as tf
import cv2
import numpy
import matplotlib.pyplot as plt
from model import build_can as model

class Model:
    def __init__(self):
        self.model = None
        
    def load_model(self,path):
        self.model = tf.keras.models.load_model(path)
        
    def predict(self,image):
        return self.model.predict(image)



class PreprocessPipeline:
    def __init__(self,path_to_model:str):
        self.model = tf.keras.models.load_model(path_to_model)
        # self.model_arch = model(input_shape = (160, 160, 3),
        #                           conv_channels=64,
        #                           out_channels=3,
        #                           name='can')
        
        
    def __preprocess_image(self,image):
        image = cv2.resize(image,(160,160))
        if image.max() > 50:
            image  = image/255
        image = image.astype('float32')
        image = image.reshape(1,160,160,3)
        return image
    
    def __postprocess_image(self,image):
        image = image.reshape(160,160,3)
        image = image * 255
        return image
    
    def process(self,image):
        # self.model_arch.load_weights('FIRSTMODEL_weight.HDF5 ')
        img = self.__preprocess_image(image)
        prediction = self.model.predict(img)
        image_output = self.__postprocess_image(prediction)
        return image_output
    
    def predict(self,image):
        return self.model.predict(image)
    
image = plt.imread('output.png')  
image = cv2.resize(image,(160,160))
plt.imshow(image)
plt.show()
pipeline = PreprocessPipeline('FIRSTMODEL.h5')
out = pipeline.process(image)
# plt.imshow(out)


# image = image*255
# image = cv2.resize(image,(150,150))
# if image.max() > 50:
#     image  = image/255
# image = image.astype('float32')
# image = image.reshape(1,150,150,3)

# prediction = pipeline.predict(image)
# prediction = prediction.reshape(150,150,3)

# plt.imshow(prediction)

out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
cv2.imwrite('test.png',out)


test = plt.imread('test.png')
plt.imshow(test)
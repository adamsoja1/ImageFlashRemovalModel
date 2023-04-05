import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt




class PreprocessPipeline:
    def __init__(self,path_to_model:str):
        self.model = tf.keras.models.load_model(path_to_model)

        
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
        image = np.array(image)
        img = self.__preprocess_image(image)
        prediction = self.model.predict(img)
        image_output = self.__postprocess_image(prediction)
        return image_output
    
    def predict(self,image):
        return self.model.predict(image)
    

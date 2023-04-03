import os
import numpy as np
import shutil
from flash_function import add_flashlight
import random
import cv2
import matplotlib.pyplot as plt

class DataPreparation:
    def __init__(self,
                 images_dir:str,
                 flashed_dir:str):
        
        self.images_dir = images_dir
        self.flashed_dir = flashed_dir
        
    def __load_image(self,
                     image_path:str):
        
        image = plt.imread(image_path)
        return image
    
    def __load_flashed_image(self,
                             image_path:str):
        image = plt.imread(image_path)
        image = cv2.resize(image,(150,150))
        radius = random.randint(5,30)
        intensity = random.randint(50,300)
        image = add_flashlight(image, radius, intensity)
        return image
        
    def __images_list(self):
        return os.listdir(self.images_dir)
    
    def save_images(self):
        images_list = self.__images_list()
        
        if self.flashed_dir not in os.listdir():
            os.mkdir('flashed') 
            
        for image in images_list:
            path = f'{self.images_dir}/{image}'
            img = self.__load_flashed_image(path)
            path_destination = f'{self.flashed_dir}/{image}'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_destination, img)
            
            


data = DataPreparation('flickr30k_images','flashed')
data.save_images()
        
        
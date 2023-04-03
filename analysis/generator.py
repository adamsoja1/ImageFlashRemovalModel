import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def Generator(path_image,path_flash,batch_size):
    files = os.listdir(path_image)
    L = len(files)
    while True:
        batch_start = 0
        batch_size_end = batch_size
        while batch_start < L:
            limit = min(batch_size_end,L)
            files_batched = files[batch_start:limit]
            #loading data
            x_train = []
            y_train = []
            for file in files_batched:
                X_train = plt.imread(f'{path_image}/{file}')
                Y_train = plt.imread(f'{path_flash}/{file}')
                X_train = cv2.resize(X_train,(160,160))
                Y_train = cv2.resize(Y_train,(160,160))
                # X_train = cv2.cvtColor(X_train, cv2.COLOR_BGR2RGB)
                
                x_train.append(X_train)
                y_train.append(Y_train)
                
                
            l = len(x_train)    
            x_train = np.array(x_train)
            y_train = np.array(y_train)
 
            
         
            
            x_train = x_train/255 
            y_train = y_train/255
            
            yield(y_train, x_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size

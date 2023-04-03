from model import model_res as model
from generator import Generator 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

PATH_IMAGE = 'flickr30k_images'
PATH_FLASH = 'flashed'
BATCH_SIZE = 20
INPUT_SIZE = (160,160,3)
OPTIMIZER  = tf.keras.optimizers.Adam(lr=0.0001)
LOSS = 'mse'
METRICS = ['accuracy',]
STEPS_PER_EPOCH = len(os.listdir(PATH_IMAGE))//BATCH_SIZE

gener = Generator(PATH_IMAGE, PATH_FLASH, BATCH_SIZE)

model = model((160, 160, 3))

model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS])
model.summary()

history = model.fit(
            gener,
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs = 20)



model.save(f'FIRSTMODEL.h5')

# model.save_weights("FIRSTMODEL_weight.HDF5 ")

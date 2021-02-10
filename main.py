gpu_instance_index = "0"
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_instance_index
import pickle
from data_generator import DataGenerator, augmentations
from model import siamese_model
from loss import loss
from metrics import *
from tensorflow.keras import optimizers
import numpy as np


train = pickle.load(open('./data/train.pkl', 'rb'))
validation = pickle.load(open('./data/validation.pkl', 'rb'))
test = pickle.load(open('./data/test.pkl', 'rb'))

images_dir = '../data/images'
input_shape = (423, 512, 3)
embedding_size = 100


batch_size = 16
train_steps_per_epoch = int(train.shape[0]/batch_size)
validation_steps_per_epoch = int(validation.shape[0]/batch_size)

train_generator = DataGenerator(train, input_shape, embedding_size, train_steps_per_epoch, images_dir, augmentations, batch_size = batch_size)
validation_generator = DataGenerator(validation, input_shape, embedding_size, validation_steps_per_epoch, images_dir, batch_size = batch_size)
num_classes = train_generator.num_classes


model = siamese_model(input_shape, num_classes, embedding_size).siamese
model.compile(loss = loss(num_classes, embedding_size), optimizer = optimizers.SGD(lr=0.001))

model.fit(train_generator,
                    validation_data = validation_generator,
                    callbacks=callbacks_list,
                    epochs=100)

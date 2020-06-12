import numpy as np
np.random.seed(2020)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import statistics
import random
import time
import tensorflow.compat.v1 as tf
import keras

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import backend as K

from sklearn.metrics import log_loss
from imageio import imread

from PIL import Image

#globals for rows and cols since we will always be doing images of the same size and they will always be 3 for RGB or 1 for Gray
rows = 60
cols = 80
RGB = 3

train_data, train_target, driver_id, unique_drivers = load_train()
test_data, test_id = load_test()

#########RMS and Cross Entrop

batch_size = 32
epochs = 10

yfull_train = dict()
yfull_test = []


unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
unique_list_valid = ['p081']
X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

print('Start Single Run')
print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_valid), len(Y_valid))
print('Train drivers: ', unique_list_train)
print('Test drivers: ', unique_list_valid)


model = create_model2()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_valid, Y_valid))

# score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
# print('Score log_loss: ', score[0])

predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
score = log_loss(Y_valid, predictions_valid)
print('Score log_loss: ', score)

# Store valid predictions
for i in range(len(test_index)):
    yfull_train[test_index[i]] = predictions_valid[i]

# Store test predictions
test_prediction = model.predict(test_data, batch_size=128, verbose=1)
yfull_test.append(test_prediction)

print('Final log_loss: {}, rows: {} cols: {} epochs: {}'.format(score, rows, cols, epochs))
info_string = 'loss_' + str(score) \
                + '_r_' + str(rows) \
                + '_c_' + str(cols) \
                + '_ep_' + str(epochs)

############RMS AND KL DIV###########


batch_size = 32
epochs = 10

yfull_train = dict()
yfull_test = []


unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
unique_list_valid = ['p081']
X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

print('Start Single Run')
print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_valid), len(Y_valid))
print('Train drivers: ', unique_list_train)
print('Test drivers: ', unique_list_valid)


model = create_model3()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_valid, Y_valid))

# score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
# print('Score log_loss: ', score[0])

predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
score = log_loss(Y_valid, predictions_valid)
print('Score log_loss: ', score)

# Store valid predictions
for i in range(len(test_index)):
    yfull_train[test_index[i]] = predictions_valid[i]

# Store test predictions
test_prediction = model.predict(test_data, batch_size=128, verbose=1)
yfull_test.append(test_prediction)

print('Final log_loss: {}, rows: {} cols: {} epochs: {}'.format(score, rows, cols, epochs))
info_string = 'loss_' + str(score) \
                + '_r_' + str(rows) \
                + '_c_' + str(cols) \
                + '_ep_' + str(epochs)

############ADAM AND KL DIV###########


batch_size = 32
epochs = 10

yfull_train = dict()
yfull_test = []


unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
unique_list_valid = ['p081']
X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

print('Start Single Run')
print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_valid), len(Y_valid))
print('Train drivers: ', unique_list_train)
print('Test drivers: ', unique_list_valid)


model = create_model4()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_valid, Y_valid))

# score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
# print('Score log_loss: ', score[0])

predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
score = log_loss(Y_valid, predictions_valid)
print('Score log_loss: ', score)

# Store valid predictions
for i in range(len(test_index)):
    yfull_train[test_index[i]] = predictions_valid[i]

# Store test predictions
test_prediction = model.predict(test_data, batch_size=128, verbose=1)
yfull_test.append(test_prediction)

print('Final log_loss: {}, rows: {} cols: {} epochs: {}'.format(score, rows, cols, epochs))
info_string = 'loss_' + str(score) \
                + '_r_' + str(rows) \
                + '_c_' + str(cols) \
                + '_ep_' + str(epochs)

############rms, crossentropy and softmax###########


batch_size = 32
epochs = 10

yfull_train = dict()
yfull_test = []


unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
unique_list_valid = ['p081']
X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

print('Start Single Run')
print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_valid), len(Y_valid))
print('Train drivers: ', unique_list_train)
print('Test drivers: ', unique_list_valid)


model = create_model5()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_valid, Y_valid))

# score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
# print('Score log_loss: ', score[0])

predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
score = log_loss(Y_valid, predictions_valid)
print('Score log_loss: ', score)

# Store valid predictions
for i in range(len(test_index)):
    yfull_train[test_index[i]] = predictions_valid[i]

# Store test predictions
test_prediction = model.predict(test_data, batch_size=128, verbose=1)
yfull_test.append(test_prediction)

print('Final log_loss: {}, rows: {} cols: {} epochs: {}'.format(score, rows, cols, epochs))
info_string = 'loss_' + str(score) \
                + '_r_' + str(rows) \
                + '_c_' + str(cols) \
                + '_ep_' + str(epochs)

############rms, kldiv and softmax###########


batch_size = 32
epochs = 10

yfull_train = dict()
yfull_test = []


unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
unique_list_valid = ['p081']
X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

print('Start Single Run')
print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_valid), len(Y_valid))
print('Train drivers: ', unique_list_train)
print('Test drivers: ', unique_list_valid)


model = create_model6()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_valid, Y_valid))

# score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
# print('Score log_loss: ', score[0])

predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
score = log_loss(Y_valid, predictions_valid)
print('Score log_loss: ', score)

# Store valid predictions
for i in range(len(test_index)):
    yfull_train[test_index[i]] = predictions_valid[i]

# Store test predictions
test_prediction = model.predict(test_data, batch_size=128, verbose=1)
yfull_test.append(test_prediction)

print('Final log_loss: {}, rows: {} cols: {} epochs: {}'.format(score, rows, cols, epochs))
info_string = 'loss_' + str(score) \
                + '_r_' + str(rows) \
                + '_c_' + str(cols) \
                + '_ep_' + str(epochs)

############adam, kldiv and softmax###########


batch_size = 32
epochs = 10

yfull_train = dict()
yfull_test = []


unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
unique_list_valid = ['p081']
X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

print('Start Single Run')
print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_valid), len(Y_valid))
print('Train drivers: ', unique_list_train)
print('Test drivers: ', unique_list_valid)


model = create_model7()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_valid, Y_valid))

# score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
# print('Score log_loss: ', score[0])

predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
score = log_loss(Y_valid, predictions_valid)
print('Score log_loss: ', score)

# Store valid predictions
for i in range(len(test_index)):
    yfull_train[test_index[i]] = predictions_valid[i]

# Store test predictions
test_prediction = model.predict(test_data, batch_size=128, verbose=1)
yfull_test.append(test_prediction)

print('Final log_loss: {}, rows: {} cols: {} epochs: {}'.format(score, rows, cols, epochs))
info_string = 'loss_' + str(score) \
                + '_r_' + str(rows) \
                + '_c_' + str(cols) \
                + '_ep_' + str(epochs)
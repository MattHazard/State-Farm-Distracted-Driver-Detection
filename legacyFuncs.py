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

def storeData(data, fPath):
    f = open(fPath, 'wb')
    pickle.dump(data, f, protocol=4)
    f.close()


def getData(fPath):
    if os.path.isfile(fPath):
        f = open(fPath, 'rb')
        data = pickle.load(f)
    else:
        print('File does not exist')
        print('Returning empty dict instead')
        data = dict()
    return data


def get_driver_data():
    dr = dict()
    path = ('driver_imgs_list.csv')
    # print(path)
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


def load_train():
    trainFile = 'train.pickle'
    if os.path.isfile(trainFile):
        X_train, y_train, driver_id, unique_drivers = getData(trainFile)
        print(unique_drivers)
        return X_train, y_train, driver_id, unique_drivers
    else:
        X_train, y_train, driver_id, unique_drivers = train_helper()
        storeData((X_train, y_train, driver_id, unique_drivers), trainFile)
        return X_train, y_train, driver_id, unique_drivers


def train_helper():
    X_train = []
    y_train = []
    driver_id = []
    start_time = time.time()
    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('train', 'c' + str(j), '*.jpg')
        # print(path)
        # return
        files = glob.glob(path)
        for fl in files:
            if RGB == 1:
                img = cv2.imread(fl, 0)
            else:
                img = cv2.imread(fl)
            # img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[os.path.basename(fl)])

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    # print(unique_drivers)

    X_train = np.array(X_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.uint8)

    X_train = X_train.reshape(X_train.shape[0], RGB, rows, cols)
    y_train = np_utils.to_categorical(y_train, 10)

    X_train = X_train.astype('float32')
    X_train /= 255

    return X_train, y_train, driver_id, unique_drivers


def load_test():
    testFile = 'test.pickle'
    if os.path.isfile(testFile):
        X_test, X_test_id = getData(testFile)
        return X_test, X_test_id
    else:
        X_test, X_test_id = test_helper()
        storeData((X_test, X_test_id), testFile)
        return X_test, X_test_id


def test_helper():
    print('Read test images')
    start_time = time.time()
    path = os.path.join('test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files) / 10)
    for fl in files:
        if RGB == 1:
            img = cv2.imread(fl, 0)
        else:
            img = cv2.imread(fl)
        X_test.append(img)
        X_test_id.append(os.path.basename(fl))
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))

    X_test = np.array(X_test, dtype=np.uint8)
    X_test = X_test.reshape(X_test.shape[0], RGB, rows, cols)

    X_test = X_test.astype('float32')
    X_test /= 255

    return X_test, X_test_id


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index
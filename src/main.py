#!/usr/bin/env python3

# Imports!
import os
import random

import cv2
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report

import model

# Config and Inits
data_dir = '/Users/skwe9/Desktop/heartlab/chest_xray/' # change directory here
size = (256, 256)

# Fucntions!
def img_2_arr(
    img_path: str,
    resize: bool = False,
    grayscale: bool = True,
    size: tuple = (256, 256),
) -> np.ndarray:

    """
    This function is responsible for opening an image, Preprocessing
    it by color or size and returning a numpy array.

    Input:
        - img_path: str, a path to the location of a image file on disk
        - resize: bool, True/False if the image is to be resized
        - grayscale: bool, True/False if image is meant to be B&W or color
        - size: tuple, a 2d tuple containing the x/y size of the image.

    Output:
        - a np.ndarray which is assosiated to the image that was input.
    """

    if grayscale:
        img_arr = cv2.imread(img_path, 0)

    else:
        img_arr = cv2.imread(img_path)
    if resize:
        img_arr = cv2.resize(img_arr, size)
    return img_arr


def create_training_datasets(data_dir: str) -> np.ndarray:
    """
    This function is responsible for creating a training dataset which
    contains images and their associated class.

    Inputs:
        - data_dir: str, which is the location where the chest x-rays are
            located.

    Outputs:
        - a np.ndarray which contains the processed image, and the class
            int, associated with that class.

    """
    # Image Loading and Preprocessing
    training_normal_img_paths = []
    training_viral_img_paths = []
    training_bact_img_paths = []

    train_dir = os.path.join(data_dir, "TRAIN") # Set directory to training images

    for cls in os.listdir(train_dir): # NORMAL or PNEUMONIA
        for img in os.listdir(os.path.join(train_dir, cls)): # all training images
            if cls == "NORMAL":
                training_normal_img_paths.append(os.path.join(train_dir, cls, img))
            elif "virus" in img:
                training_viral_img_paths.append(os.path.join(train_dir, cls, img))
            elif "bacteria" in img: #stop it from finding desktop.ini file
                training_bact_img_paths.append(os.path.join(train_dir, cls, img))

    # 0 for normal, 1 for bacterial and 2 for viral
    training_dataset = (
        [
            [img_2_arr(path, grayscale=True, resize=True, size=size), 0]
            for path in training_normal_img_paths
        ]
        + [
            [img_2_arr(path, grayscale=True, resize=True, size=size), 1]
            for path in training_bact_img_paths
        ]
        + [
            [img_2_arr(path, grayscale=True, resize=True, size=size), 2]
            for path in training_viral_img_paths
        ]
    )

    return np.array(training_dataset, dtype="object")


def create_testing_datasets(data_dir: str) -> np.ndarray:
    """
    This function is responsible for creating a testing dataset which
    contains images and their associated class.

    Inputs:
        - data_dir: str, which is the location where the chest x-rays are
            located.

    Outputs:
        - a np.ndarray which contains the processed image, and the class
            int, associated with that class.

    """
    # Image Loading and Preprocessing
    testing_normal_img_paths = []
    testing_viral_img_paths = []
    testing_bact_img_paths = []

    test_dir = os.path.join(data_dir, "TEST") # Set directory to testing images

    for cls in os.listdir(test_dir): # NORMAL or PNEUMONIA
        for img in os.listdir(os.path.join(test_dir, cls)): # all testing images
            if cls == "NORMAL":
                testing_normal_img_paths.append(os.path.join(test_dir, cls, img))
            elif "virus" in img:
                testing_viral_img_paths.append(os.path.join(test_dir, cls, img))
            elif "bacteria" in img: # stop it from finding desktop.ini file
                testing_bact_img_paths.append(os.path.join(test_dir, cls, img))

    # 0 for normal, 1 for bacterial and 2 for viral
    testing_dataset = (
        [
            [img_2_arr(path, grayscale=True, resize=True, size=size), 0]
            for path in testing_normal_img_paths
        ]
        + [
            [img_2_arr(path, grayscale=True, resize=True, size=size), 1]
            for path in testing_bact_img_paths
        ]
        + [
            [img_2_arr(path, grayscale=True, resize=True, size=size), 2]
            for path in testing_viral_img_paths
        ]
    )

    return np.array(testing_dataset, dtype="object")


def main():

    # get training and testing dataset
    training_dataset = create_training_datasets(data_dir)
    testing_dataset = create_testing_datasets(data_dir)

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    #split data into image and label
    for image, label in training_dataset:
        X_train.append(image)
        y_train.append(label)

    for image, label in testing_dataset:
        X_test.append(image)
        y_test.append(label)

    # normalise the data
    X_test = np.array(X_test) / 255
    X_train = np.array(X_train) / 255

    y_train = to_categorical(y_train, 3)
    y_test = to_categorical(y_test, 3)

    # Resize data for deep learning 
    X_train = X_train.reshape(-1,256,256,1)
    y_train = np.array(y_train)

    X_test = X_test.reshape(-1,256,256,1)
    y_test = np.array(y_test)

    datagen = ImageDataGenerator(
        zoom_range = 0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
    )

    datagen.fit(X_train)

    class_weights = {0:2.626, 1:1.3528, 2:3.834} # total samples / n class samples = weight or bias for each output

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience = 2,
                                            verbose=1,
                                            factor=0.3,
                                            min_lr=0.000001)

    history = model.model_zero().fit(datagen.flow(X_train, y_train, batch_size = 32),
                    epochs = 15,
                    shuffle= True,
                    validation_data = datagen.flow(X_test, y_test),
                    callbacks = [learning_rate_reduction],
                    class_weight=class_weights
                   )

    print(history)

    print("Loss of the model is - " , model.model_zero().evaluate(X_test,y_test)[0])
    print("Accuracy of the model is - " , model.model_zero().evaluate(X_test,y_test)[1]*100 , "%")


if __name__ == "__main__":

    main()

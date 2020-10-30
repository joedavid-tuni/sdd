import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from utils import mask2contour
from keras import backend as K
import segmentation_models as sm
from os import listdir
from PIL import Image
import argparse
import sys


def dice_coef(y_true, y_pred, smooth=1):
    """Calculates the dice coefficient

    Args:
        y_true: pixel truth values
        y_pred: pixel predicted values
        smooth: smooth parameter to smooth the loss function

    Returns:
        (float): Returns the dice coefficient
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_model():
    """Gets the model used for training the dataset

    Returns:
        model: the model used for training
        preprocess: the preprocessing model
    """

    preprocess = sm.get_preprocessing('resnet34')
    model = sm.Unet('resnet34', input_shape=(480, 640, 3), classes=2, activation='sigmoid')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    return model, preprocess

def train(path, save_file):

    """ Trains the model

        Args:
            path (str): root directory containing the training dataset
            save_file (str): directory to save the trained model for use at a later stage
    """

    model, preprocess = get_model()
    train_batches = DataGenerator(N=69, no_of_classes=2, path = path, shuffle=True, preprocess=preprocess, purpose='train', batch='train')
    valid_batches = DataGenerator(N=69, no_of_classes=2, path = path, preprocess=preprocess, purpose ='train', batch='test')
    model.fit_generator(train_batches, validation_data=valid_batches, epochs=1, verbose=1)
    model.save(save_file)

def load_and_visualize(path):
    """Visualizes the training data with its masks

    Args:
          path (str): root directory containing the images to visualize
    """

    dg = DataGenerator(N=69, path=path, batch_size=6, no_of_classes=2, img_size=(640, 480), purpose='visualize')
    for batch_idx, (image_names, X, Y) in enumerate(dg):  # loop batches one by one
        fig = plt.figure(figsize=(25, 25))
        for idx, (img, masks) in enumerate(zip(X, Y)):  # loop of images
            # print(batch_idx, image_names[idx])
            for m in range(2):  # loop different defects
                mask = masks[:, :, m]
                mask = mask2contour(mask, width=2)
                if m == 0:  # yellow
                    img[mask == 1, 0] = 235
                    img[mask == 1, 1] = 235
                elif m == 1:
                    img[mask == 1, 1] = 210  # green
                elif m == 2:
                    img[mask == 1, 2] = 255  # blue
            plt.axis('off')
            fig.add_subplot(2, 3, idx+1)
            plt.imshow(img/255.0)
            plt.title(image_names[idx])
        plt.show()


def main():
    path = '/home/joe/Documents/steel_defect_detection/resized_images/'
    # load_and_visualize(path)
    train(path,'./model.h5')


if __name__ == '__main__':
    main()









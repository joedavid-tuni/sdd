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





def load_and_visualize(path):
    """Visualizes the training data with its masks

    Args:
          path (str): root directory containing the images to visualize
    """

    dg = DataGenerator(N=69, path=path, batch_size=6, no_of_classes=3, img_size=(640, 480))
    for batch_idx, (image_names, X, Y) in enumerate(dg):  # loop batches one by one
        fig = plt.figure(figsize=(25, 25))
        for idx, (img, masks) in enumerate(zip(X, Y)):  # loop of images
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
    load_and_visualize('/home/joe/Documents/steel_defect_detection/resized_images/')


if __name__ == '__main__':
    main()









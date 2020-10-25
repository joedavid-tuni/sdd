import numpy as np
import os
import cv2
from tqdm import tqdm
import logging
from PIL import Image
from pathlib import Path


def rle2mask(rle_string, shape=(1600, 256)):
    """ Converts run-length-encoded mask to numpy array of image dimensions

    Args:
        mask_rle (str): run-length encoding as string formated (start length)
        shape (tuple): (height,width) of array to return

    Returns:
        (ndarray) numpy mask array (1 - mask, 0 - background)
    """
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T # Transposed Needed to accommodate with receiving array dimensions


def mask2contour(mask, width=3):
    """ Converts mask to contour

    Args:
        mask (ndarray): mask to be converted
        width (int): width of contour

    Returns:
        (ndarray): numpy contour array
    """
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:, width:], np.zeros((h, width))], axis=1)
    mask2 = np.logical_xor(mask, mask2)
    mask3 = np.concatenate([mask[width:, :], np.zeros((width, w))], axis=0)
    mask3 = np.logical_xor(mask, mask3)
    return np.logical_or(mask2, mask3)


def rename_images(dir):
    """ This function renames images in a directory that has images categorized in folders that have the name of the
    class. The function renames all the images as based on the class and an image count with the following format:

    <CLass Name>_Image_<Image Count>

    For example the first image in the folder Cat will be renamed as Cat_Image_1 and so on.

    Note that the original the images will be overwritten


    Args:
        dir: directory containing class-named folders with images

    Returns: None

    """

    image_file_extenstions = ['bmp', 'jpg', 'png', 'jpeg']

    image_counter = 1
    for (dirname, dirs, files) in os.walk(dir):
        for file in files:
            ext = file.split('.')[-1]
            if ext in image_file_extenstions:
                os.rename(dirname + "/" + file, dirname + "/" + dirname.rsplit('/')[-1] + "_Image_" + str(image_counter) + "." + ext)
                image_counter = image_counter + 1
            else:
                print("Non (supported) image file type detected, skipping")
    print(image_counter, ' images processed')


def resize_and_format(dir, factor=0.5, change_format=False, new_format='png'):
    """

    This function resizes the images in a directory and saves them in another directory called resized images
    at the same level of the supplied directory. The function also changes the format if necessary.

    Note: A log file logs the resolutions of the before and after images.

    Args:
        dir (str): the directory containing the images
        factor (float): the factor by which the size must be resized, default = 0.5 (half)
        change_format (bool): Boolean value indicating if there needs to be a change in format in addition to resizing
        new_format (str): the new format to change to, default = 'png'

    Returns:
        resize_dir(str): the directory containing the resized images.

    """
    resize_parent_dir = str(Path(dir).parents[0])
    resize_dir = resize_parent_dir + "/resized_images"

    try:
        os.mkdir(resize_dir)
    except OSError:
        print("Creation of the directory %s failed" % resize_dir)
    no_of_images = len(os.listdir(dir))
    print("\n", no_of_images, 'images found in directory... Resizing')

    for img_filename in tqdm(os.listdir(dir)):
        img_dir = dir + "/" + img_filename
        image = cv2.imread(img_dir)
        dim = (int(image.shape[1] * factor), int(image.shape[0] * factor))
        logging.debug("First Image has a resolution of: {} x {}".format(image.shape[1], image.shape[0]))
        logging.debug("Resizing  to resolution {} x {}".format(dim[0], dim[1]))
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(resize_dir + "/" + img_filename, resized_image)

    if change_format:
        print('Changing Image format to {}'.format(new_format))

        for img_filename in tqdm(os.listdir(resize_dir)):
            img_dir = resize_dir + "/" + img_filename
            img = Image.open(img_dir)
            img.save(img_dir.rsplit('.')[0] + '.' + new_format, new_format)
            os.remove(img_dir)
    return resize_dir

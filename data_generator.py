import numpy as np
import keras
from utils import rle_decode
import matplotlib.pyplot as plt
import ijson


class DataGenerator(keras.utils.Sequence):
    """ Generates real-time data feed to model in batches

    """

    def __init__(self, N, path, no_of_classes, img_size=(640, 480), batch_size=16, shuffle=False, preprocess=None):
        """ Initializes the DataGenerator Object

        Args:
            shuffle (bool) : whether to shuffle the order in which data is fed to the model
            batch_size (int): size of batch that arrives as data feed
            preprocess : pre-processing BACKBONE
            path (str) : root directory containing the training and testing dataset
            img_size (tuple): size of the image to be provided in the data feed

        """

        super().__init__()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.N = N
        self.path = path
        self.img_size = img_size
        self.on_epoch_end()
        self.indexes = np.arange(self.N)
        self.no_of_classes = no_of_classes

        annotations_PREFIX = "images.item"
        f = open('ICH.json')

        self.annotations = ijson.items(f, annotations_PREFIX)


    def __len__(self):
        """Calculates number of batch in the Sequence.

        Returns:
            (int): The number of batches in the Sequence.
        """
        return int(np.floor(self.N / self.batch_size))

    def on_epoch_end(self):
        """ A method called at the end of every epoch that shuffles the data if parameter shuffle = True.

        """

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def fetch_data(self):
        for idx, annotation_object in enumerate(self.annotations):
            yield idx, annotation_object



    def __getitem__(self, index):
        """Generates one batch of data at position 'index'.

        Note: index=0 for batch 1, 2 for batch 2 and so on..

        Arguments:
            index (int): position of the batch in the Sequence.
        Returns:
            A batch.
            X (ndarray): array containing the image
                        4D array of size: batch_size x img_size[0] x img_size[1] x 3 (RGB)
            Y (ndarray): array containing the masks for corresponding images in X
                        4D array of size: batch_size x img_size[0] x img_size[1] x 4 (number of defect classes)

        Note: If subset =' train', both the images along with its masks is returned. This is essentially the information
        contained in the train.csv file. If subset = 'test', only the images in the test_images folder is returned
        """
        # (batch size, image height (rows), image width (columns), number of channels (RGB=3))
        X = np.zeros((self.batch_size, self.img_size[1], self.img_size[0], 3), dtype=np.float32)

        # (batch size, image height (rows), image width (columns), number of (defect) classes = 3)
        Y = np.zeros((self.batch_size, self.img_size[1], self.img_size[0], self.no_of_classes), dtype=np.int8)

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        class_map = {'scratch': 0 , 'blister': 1, 'dent': 2}
        image_names = []
        for idx, annotation_object in self.fetch_data():

            image_name = annotation_object["image_name"]
            image_names.append(image_name)
            X[idx, ] = plt.imread(self.path + image_name)
            masks_rle = []
            defect_classes = []
            bboxes = []
            masks = np.zeros((self.img_size[1], self.img_size[0], self.no_of_classes), dtype=np.int8)

            for label in annotation_object["labels"]:
                mask_rle = label["mask"]
                masks_rle.append(mask_rle)
                defect_classes.append(label["class_name"])
                bboxes.append(label["bbox"])

            # mapping defect classes to integers based on class_map dict
            defect_classes = list(map(class_map.get, defect_classes))

            for defect_class, mask, bbox in zip(defect_classes, masks_rle, bboxes):

                # if brush tool is used
                if mask is not None:
                    x_min, y_min, x_max, y_max = bbox
                    x_diff = x_max - x_min
                    y_diff = y_max - y_min
                    _mask = rle_decode(mask, (x_diff,y_diff))

                    defect_mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.int8)

                    for index, x in np.ndenumerate(_mask):
                        if x == 1:
                            defect_mask[index[0] + y_min, index[1] + x_min] = 1

                    masks[:, :, defect_class] = np.logical_or(masks[:, :, defect_class], defect_mask)

            for defect_class in defect_classes:
                Y[idx, :, :, defect_class] = masks[:, :, defect_class]

            if idx == self.batch_size-1:
                if self.preprocess is not None: X = self.preprocess(X)
                return image_names, X, Y
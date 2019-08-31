from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow.python as tf
import tensorflow.python.keras.backend as K
import numpy as np

from yolo.utils import get_random_data
from yolo.model import preprocess_true_boxes, yolo_body, yolo_loss


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, attribute, anchors, input_shape, batch_size=8, shuffle=True):
        self.data = data
        self.indexes = np.arange(len(self.data))
        self.attribute = attribute
        self.anchors = anchors
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch_index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = [self.data[k] for k in batch_index]
        x, y = self.data_generator(batch_data)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generator(self, batch_data):
        num_seen = self.attribute.shape[0]
        attribute = np.tile(np.expand_dims(self.attribute, 0), (self.batch_size, 1, 1))

        image_data = []
        box_data = []
        for data in batch_data:
            image, box = get_random_data(data, self.input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, num_seen)
        return [image_data, *y_true, attribute], np.zeros(self.batch_size)


def load_anchors():
    anchors_path = "data/yolo_anchors.txt"
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def train():
    annotation_path = "ZJLAB_ZSD_2019/instances_train.json"
    attribute_path = "ZJLAB_ZSD_2019/seen_embeddings_Bert.json"
    log_dir = "logs"
    anchors_path = "data/yolo_anchors.txt"
    weights_path = ""



if __name__ == '__main__':
    pass

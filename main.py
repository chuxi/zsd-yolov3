from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub

import pathlib
import random
import os
import json

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_INPUT_SIZE = [416, 416]


def load_and_preprocess_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_INPUT_SIZE)
    img = img / 255.0
    return img


def prepare_img(data_dir):
    data_dir_path = pathlib.Path(data_dir)
    img_files = list(data_dir_path.glob('*.jpg'))
    img_files = [str(path) for path in img_files]
    random.shuffle(img_files)

    path_ds = tf.data.Dataset.from_tensor_slices(img_files)
    img_ds = path_ds.map(load_and_preprocess_img, num_parallel_calls=AUTOTUNE)

    return img_ds


def load_instances_train(instances_train_file):
    with open(instances_train_file, mode='r') as f:
        obj = json.load(f)

        for i in range(3):
            print(obj['annotations'][i])


def load_training_data():
    train_img_dir = "./ZJLAB_ZSD_2019/train"
    instances_train_file = "./ZJLAB_ZSD_2019/instances_train.json"
    with open(instances_train_file, mode='r') as f:
        instances_json = json.load(f)
    categories = dict((item["id"], item["name"]) for item in instances_json["categories"])
    images = dict((item["id"], item["file_name"]) for item in instances_json["images"])
    annotations = dict((item["id"], item["file_name"]) for item in instances_json["annotations"])

    pass


def load_model():
    detector_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.Module(detector_url)
    img_url = "./ZJLAB_ZSD_2019/train/img_00000000.jpg"
    img = load_and_preprocess_img(img_url)
    detector_output = detector(img, as_dict=True)
    print(detector_output)




if __name__ == '__main__':
    os.environ['https_proxy'] = "https://localhost:8123"
    load_model()
    pass

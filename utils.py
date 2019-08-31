import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_img_dir = "./ZJLAB_ZSD_2019/train/"
instances_train_file = "./ZJLAB_ZSD_2019/instances_train.json"

def load_training_data(input_shape=(416,416)):
    with open(instances_train_file, mode='r') as f:
        instances_json = json.load(f)
    categories = dict((item["id"], item["name"]) for item in instances_json["categories"])
    images = dict((item["id"], item["file_name"]) for item in instances_json["images"])
    annotations = dict((item["id"], item) for item in instances_json["annotations"])
    print("categories: ", instances_json["categories"][0])
    print("images: ", instances_json["images"][0])
    print("annotations: ", instances_json["annotations"][0])
    num_seen_classes = len(categories)
    print("num seen classes: ", num_seen_classes)
    img_data = []
    box_data = []
    label_data = []
    for item in instances_json["annotations"]:
        image_file_name = "img_%08d.jpg" % item["image_id"]
        image_box = item["bbox"]
        # resize image and bbox data
        raw_image = tf.io.read_file(train_img_dir + image_file_name)
        raw_image = tf.image.decode_jpeg(raw_image, channels=3)
        raw_width, raw_height = raw_image.shape
        input_width, input_height = input_shape
        scale_width, scale_height = input_width / raw_width, input_height / raw_height


def _load_image_from_file(image_file_path, input_shape=(416,416)):
    img = tf.io.read_file(image_file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, input_shape)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def display_annotation(annotation):
    print("annotation: ", annotation)
    image_file_name = "img_%08d.jpg" % annotation["image_id"]
    img = tf.io.read_file(train_img_dir + image_file_name)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    height, width, _ = img.shape
    print("image shape: ", img.shape)
    bbox = tf.cast(annotation["bbox"], tf.float32)
    print("bbox value: ", bbox)
    bbox = [bbox[1]/height, bbox[0]/width,
            (bbox[1] + bbox[3])/height, (bbox[0] + bbox[2] + 1)/width]
    print("final bbox: ", bbox)
    image = tf.image.draw_bounding_boxes(tf.expand_dims(img, 0),
                                 tf.expand_dims(tf.expand_dims(bbox, 0), 0), [[0,0,0]])
    plt.imshow(image[0])
    plt.show()


def _process_bbox(bbox, raw_shape, input_shape=(416,416)):
    """

    :param bbox: bbox values from
    :param raw_shape:
    :param input_shape:
    :return:
    """
    raw_w, raw_h = raw_shape
    in_w, in_h = input_shape
    scale_w, scale_h = input_shape
    b_x, b_y, b_w, b_h = bbox


if __name__ == '__main__':
    with open(instances_train_file, mode='r') as f:
        instances_json = json.load(f)
        display_annotation(instances_json["annotations"][0])
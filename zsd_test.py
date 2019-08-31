"""
Test zero-shot YOLO detection model on unseen classes.
"""
import collections
import json
import os

import numpy as np
from PIL import Image
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from tqdm import tqdm

from yolo.model import yolo_body, yolo_eval
from yolo.utils import letterbox_image

from zsd_train import get_embeddings, get_categories, get_anchors

# seen_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'cat', 'chair', 'cow', 'diningtable',
#                 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'tvmonitor']
#
# unseen_classes = ['car', 'dog', 'sofa', 'train']
#
# total_classes = seen_classes + unseen_classes


class YOLO(object):
    def __init__(self):
        self.weight_path = 'logs/voc_04/trained_weights_sigmoid.h5'
        self.anchors_path = 'data/yolo_anchors.txt'
        self.attribute_path = 'model_data/attributes.npy'

        self.predict_dir = 'predicted/'
        self.score = 0.001
        self.iou = 0.5
        self.num_seen = 80
        self.anchors = get_anchors(self.anchors_path)
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.seen_classes = get_categories("ZJLAB_ZSD_2019/instances_train.json")
        self.unseen_classes = self._get_unseen_categories()
        self.boxes, self.scores, self.classes = self.generate()

    def get_unseen_embeddings(self):
        with open("unseen_GloVe.json") as f:
            embeddings = json.load(f)
        embeddings = dict((self.unseen_classes.index(k), v) for k, v in embeddings.items())
        # order by class id
        embeddings = collections.OrderedDict(sorted(embeddings.items()))
        return np.array([v[0] for k,v in embeddings.items()])

    def _get_unseen_categories(self):
        with open("clsname2id_20k.json") as f:
            categories = json.load(f)
        id2clsname = dict((int(v) - 20, k) for k, v in categories.items())
        unseen_classes = collections.OrderedDict(sorted(id2clsname.items()))
        return list(unseen_classes.values())

    def get_attribute(self):
        unseen_embeddings = self.get_unseen_embeddings()

        # load seen_classes
        embeddings = get_embeddings("ZJLAB_ZSD_2019/seen_embeddings_GloVe.json", self.seen_classes)
        return np.concatenate((embeddings, unseen_embeddings), axis=0)

    def generate(self):
        model_path = os.path.expanduser(self.weight_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)

        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), self.num_seen, num_anchors // 3)
        self.yolo_model.load_weights(self.weight_path, by_name=True)
        print('{} model, anchors and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        attribute = self.get_attribute()
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, self.num_seen,
                                           attribute, self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image_path):
        image = Image.open(image_path)

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        image_name = image_path.split('/')[-1].split('.')[0]
        with open(os.path.join(self.predict_dir, image_name + '.txt'), 'w') as f:
            f.write(image_name)
            for i, c in enumerate(out_classes):
                class_name = self.unseen_classes[c]
                confidence = out_scores[i]
                box = out_boxes[i]

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                f.write('{} {} {} {} {} {}\n'.format(left, top, right, bottom, confidence, class_name))

    def close_session(self):
        self.sess.close()


def _main():
    test_path = 'data/test.txt'

    yolo = YOLO()
    with open(test_path) as rf:
        test_img = rf.readlines()
    test_img = [c.strip() for c in test_img]

    for img in tqdm(test_img):
        img_path = img.split()[0]
        yolo.detect_image(img_path)
    K.clear_session()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    _main()

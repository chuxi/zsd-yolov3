from __future__ import absolute_import, division, print_function, unicode_literals

import abc

import six as six
import tensorflow as tf

import rcnn.standard_fields as fields


class DetectionModel(six.with_metaclass(abc.ABCMeta, tf.Module)):
    def __init__(self, num_classes):
        super().__init__()
        self._num_classes = num_classes
        self._groundtruth_lists = {}

    @property
    def num_classes(self):
        return self._num_classes

    def groundtruth_lists(self, field):
        if field not in self._groundtruth_lists:
            raise RuntimeError('Groundtruth tensor {} has not been provided'.format(
                field))
        return self._groundtruth_lists[field]

    @abc.abstractmethod
    def preprocess(self, inputs):
        """

        :param
            inputs:  a [batch, height_in, width_in, channels] float32 tensor
                representing a batch of images with values between 0 and 255.0.

        :return:
            preprocessed_inputs: a [batch, height_out, width_out, channels] float32
                tensor representing a batch of images.
            true_image_shapes: int32 tensor of shape [batch, 3] where each row is
                of the form [height, width, channels] indicating the shapes
                of true images in the resized images, as resized images can be padded
                with zeros.
        """
        pass

    @abc.abstractmethod
    def predict(self, preprocessed_inputs, true_image_shapes):
        pass

    @abc.abstractmethod
    def postprocess(self, prediction_dict, true_image_shapes, **params):
        pass

    @abc.abstractmethod
    def loss(self, prediction_dict, true_image_shapes):
        pass

    def provide_groundtruth(self,
                            groundtruth_boxes_list,
                            groundtruth_classes_list,
                            groundtruth_masks_list=None,
                            groundtruth_keypoints_list=None,
                            groundtruth_weights_list=None,
                            groundtruth_confidences_list=None,
                            groundtruth_is_crowd_list=None,
                            is_annotated_list=None):
        self._groundtruth_lists[fields.BoxListFields.boxes] = groundtruth_boxes_list
        self._groundtruth_lists[fields.BoxListFields.classes] = groundtruth_classes_list
        if groundtruth_weights_list:
            self._groundtruth_lists[fields.BoxListFields.weights] = groundtruth_weights_list
        if groundtruth_confidences_list:
            self._groundtruth_lists[fields.BoxListFields.confidences] = groundtruth_confidences_list
        if groundtruth_masks_list:
            self._groundtruth_lists[fields.BoxListFields.masks] = groundtruth_masks_list
        if groundtruth_keypoints_list:
            self._groundtruth_lists[fields.BoxListFields.keypoints] = groundtruth_keypoints_list
        if groundtruth_is_crowd_list:
            self._groundtruth_lists[fields.BoxListFields.is_crowd] = groundtruth_is_crowd_list
        if is_annotated_list:
            self._groundtruth_lists[fields.InputDataFields.is_annotated] = is_annotated_list

    @abc.abstractmethod
    def regularization_losses(self):
        pass

    @abc.abstractmethod
    def restore_map(self, fine_tune_checkpoint_type='detection'):
        pass

    @abc.abstractmethod
    def updates(self):
        pass

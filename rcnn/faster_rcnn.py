import abc

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Lambda

from rcnn import model

_UNINITIALIZED_FEATURE_EXTRACTOR = '__uninitialized__'


class FasterRCNNFeatureExtractor(object):
    """faster rcnn feature extractor"""
    def __init__(self,
                 is_training,
                 first_stage_features_stride,
                 batch_norm_trainable=False,
                 weight_decay=0.0):
        self._is_training = is_training
        self._first_stage_features_stride = first_stage_features_stride
        self._train_batch_norm = (batch_norm_trainable and is_training)
        self._weight_decay = weight_decay

    @abc.abstractmethod
    def preprocess(selfself, resized_inputs):
        pass

    @abc.abstractmethod
    def get_proposal_feature_extractor_model(self, name):
        pass

    @abc.abstractmethod
    def get_box_classifier_feature_extractor_model(self, name):
        pass

    def restore_from_classification_checkpoint_fn(self,
                                                  first_stage_feature_extractor_scope,
                                                  second_stage_feature_extractor_scope):
        variables_to_restore = {}
        for variable in tf.Module.variables:
            pass


class FasterRCNNModel(model.DetectionModel):

    def __init__(self, is_training, num_classes, image_resizer_fn, feature_extractor, number_of_stages,
                 first_stage_anchor_generator, first_stage_target_assigner, first_stage_atrous_rate,
                 first_stage_box_predictor_arg_scope_fn, first_stage_box_predictor_kernel_size,
                 first_stage_box_predictor_depth, first_stage_minibatch_size, first_stage_sampler,
                 first_stage_non_max_suppression_fn, first_stage_max_proposals, first_stage_localization_loss_weight,
                 first_stage_objectness_loss_weight, crop_and_resize_fn, initial_crop_size, maxpool_kernel_size,
                 maxpool_stride, second_stage_target_assigner, second_stage_mask_rcnn_box_predictor,
                 second_stage_batch_size, second_stage_sampler, second_stage_non_max_suppression_fn,
                 second_stage_score_conversion_fn, second_stage_localization_loss_weight,
                 second_stage_classification_loss_weight, second_stage_classification_loss,
                 second_stage_mask_prediction_loss_weight=1.0, hard_example_miner=None, parallel_iterations=16,
                 add_summaries=True, clip_anchors_to_image=False, use_static_shapes=False, resize_masks=True,
                 freeze_batchnorm=False):
        super().__init__(num_classes)

        self._is_training = is_training
        self._image_resizer_fn = image_resizer_fn
        self._resize_masks = resize_masks
        self._feature_extractor = feature_extractor

        # delay building feature extractor
        self._feature_extractor_for_proposal_features = (_UNINITIALIZED_FEATURE_EXTRACTOR)
        self._feature_extractor_for_box_classifier_features = (_UNINITIALIZED_FEATURE_EXTRACTOR)

        self._number_of_stages = number_of_stages
        self._proposal_target_assigner = first_stage_target_assigner
        self._detector_target_assigner = second_stage_target_assigner
        self._box_coder = self._proposal_target_assigner.box_coder

        self._first_stage_anchor_generator = first_stage_anchor_generator
        self._first_stage_atrous_rate = first_stage_atrous_rate
        self._first_stage_box_predictor_depth = first_stage_box_predictor_depth
        self._first_stage_box_predictor_kernel_size = (first_stage_box_predictor_kernel_size)
        self._first_stage_minibatch_size = first_stage_minibatch_size
        self._first_stage_sampler = first_stage_sampler
        # isinstance(first_stage_box_predictor_arg_scope_fn,
        #                   hyperparams_builder.KerasLayerHyperparams)
        num_anchors_per_location = (self._first_stage_anchor_generator.num_anchors_per_location())
        if num_anchors_per_location != 1:
            raise ValueError('anchor_generator is expected to generate anchors '
                             'corresponding to a single feature map.')
        conv_hyperparams = (first_stage_box_predictor_arg_scope_fn)
        self._first_stage_box_predictor_first_conv = (
            Sequential([
                Conv2D(self._first_stage_box_predictor_depth,
                       kernel_size=[self._first_stage_box_predictor_kernel_size,
                                    self._first_stage_box_predictor_kernel_size],
                       dilation_rate=self._first_stage_atrous_rate,
                       padding='SAME',
                       name='RPNConv',
                       **conv_hyperparams.params()),
                conv_hyperparams.build_batch_norm(
                    (self._is_training and not freeze_batchnorm),
                    name = 'RPNActivation'),
                Lambda(tf.nn.relu6, name='RPNActivation')
            ], name='FirstStageRPNFeatures'))
        self._first_stage_box_predictor = (
            # box_predictor_builder
        )




class InputDataFields(object):
    image = 'image'
    image_additional_channels = 'image_additional_channels'
    original_image = 'original_image'
    original_image_spatial_shape = 'original_image_spatial_shape'
    key = 'key'
    source_id = 'source_id'
    filename = 'filename'
    groundtruth_image_classes = 'groundtruth_image_classes'
    groundtruth_image_confidences = 'groundtruth_image_confidences'
    groundtruth_boxes = 'groundtruth_boxes'
    groundtruth_classes = 'groundtruth_classes'
    groundtruth_confidences = 'groundtruth_confidences'
    groundtruth_label_types = 'groundtruth_label_types'
    groundtruth_is_crowd = 'groundtruth_is_crowd'
    groundtruth_area = 'groundtruth_area'
    groundtruth_difficult = 'groundtruth_difficult'
    groundtruth_group_of = 'groundtruth_group_of'
    proposal_boxes = 'proposal_boxes'
    proposal_objectness = 'proposal_objectness'
    groundtruth_instance_masks = 'groundtruth_instance_masks'
    groundtruth_instance_boundaries = 'groundtruth_instance_boundaries'
    groundtruth_instance_classes = 'groundtruth_instance_classes'
    groundtruth_keypoints = 'groundtruth_keypoints'
    groundtruth_keypoint_visibilities = 'groundtruth_keypoint_visibilities'
    groundtruth_label_weights = 'groundtruth_label_weights'
    groundtruth_weights = 'groundtruth_weights'
    num_groundtruth_boxes = 'num_groundtruth_boxes'
    is_annotated = 'is_annotated'
    true_image_shape = 'true_image_shape'
    multiclass_scores = 'multiclass_scores'


class DetectionResultFields(object):
    source_id = 'source_id'
    key = 'key'
    detection_boxes = 'detection_boxes'
    detection_scores = 'detection_scores'
    detection_multiclass_scores = 'detection_multiclass_scores'
    detection_features = 'detection_features'
    detection_classes = 'detection_classes'
    detection_masks = 'detection_masks'
    detection_boundaries = 'detection_boundaries'
    detection_keypoints = 'detection_keypoints'
    num_detections = 'num_detections'
    raw_detection_boxes = 'raw_detection_boxes'
    raw_detection_scores = 'raw_detection_scores'
    detection_anchor_indices = 'detection_anchor_indices'


class BoxListFields(object):
    boxes = 'boxes'
    classes = 'classes'
    scores = 'scores'
    weights = 'weights'
    confidences = 'confidences'
    objectness = 'objectness'
    masks = 'masks'
    boundaries = 'boundaries'
    keypoints = 'keypoints'
    keypoint_heatmaps = 'keypoint_heatmaps'
    is_crowd = 'is_crowd'


class TfExampleFields(object):
    image_encoded = 'image/encoded'
    image_format = 'image/format'  # format is reserved keyword
    filename = 'image/filename'
    channels = 'image/channels'
    colorspace = 'image/colorspace'
    height = 'image/height'
    width = 'image/width'
    source_id = 'image/source_id'
    image_class_text = 'image/class/text'
    image_class_label = 'image/class/label'
    object_class_text = 'image/object/class/text'
    object_class_label = 'image/object/class/label'
    object_bbox_ymin = 'image/object/bbox/ymin'
    object_bbox_xmin = 'image/object/bbox/xmin'
    object_bbox_ymax = 'image/object/bbox/ymax'
    object_bbox_xmax = 'image/object/bbox/xmax'
    object_view = 'image/object/view'
    object_truncated = 'image/object/truncated'
    object_occluded = 'image/object/occluded'
    object_difficult = 'image/object/difficult'
    object_group_of = 'image/object/group_of'
    object_depiction = 'image/object/depiction'
    object_is_crowd = 'image/object/is_crowd'
    object_segment_area = 'image/object/segment/area'
    object_weight = 'image/object/weight'
    instance_masks = 'image/segmentation/object'
    instance_boundaries = 'image/boundaries/object'
    instance_classes = 'image/segmentation/object/class'
    detection_class_label = 'image/detection/label'
    detection_bbox_ymin = 'image/detection/bbox/ymin'
    detection_bbox_xmin = 'image/detection/bbox/xmin'
    detection_bbox_ymax = 'image/detection/bbox/ymax'
    detection_bbox_xmax = 'image/detection/bbox/xmax'
    detection_score = 'image/detection/score'

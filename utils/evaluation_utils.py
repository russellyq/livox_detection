import numpy as np


def get_batch_statistics_rotated_bbox(outputs, target, iou_threshold):
    batch_catrix = []
    for sample_id in range(len(outputs)):
        if outputs[sample_id] is None: continue
        output = outputs[sample_id]
        
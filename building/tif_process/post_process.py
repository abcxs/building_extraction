# -*- encoding: utf-8 -*-
#__author:"zfp"
#data:2019/6/18

import cv2
import numpy as np
import os
import pickle
import torch
from utils import get_gpus, nms_bbox, nms_rbox, rbbox_iou, remove_small_bboxes


class BoxList(object):
    def __init__(self, bboxes):
        self.bboxes = bboxes
        self.extra_fields = {}
    
    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data
    
    def get_field(self, field):
        return self.extra_fields[field]

    def fields(self):
        return list(self.extra_fields.keys())

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        boxlist = BoxList(self.bboxes[item])
        for k, v in self.extra_fields.items():
            boxlist.add_field(k, v[item])
        return boxlist

def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList into a single BoxList
    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bboxes for bbox in bboxes], dim=0))

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes

def save(output_dir, boxes, poly, mask):
    with open(os.path.join(output_dir, 'boxes.pkl'), 'wb') as f:
        pickle.dump(boxes, f)
    with open(os.path.join(output_dir, 'poly.pkl'), 'wb') as f:
        pickle.dump(poly, f)
    with open(os.path.join(output_dir, 'mask.pkl'), 'wb') as f:
        pickle.dump(mask, f)

def post_process(output_dir, cfg, width, height):
    gpus = get_gpus()

    nms_size = cfg.nms_size
    nms_overlap_size = cfg.nms_overlap_size
    nms_threshold = cfg.bbox_nms_threshold
    nms_r_threshold = cfg.rbox_nms_threshold

    CATEGORIES = cfg.CATEGORIES

    if nms_size <= 0:
        nms_size = max(width, height)
        nms_overlap_size = 0

    gpu_id = gpus[0]

    result_boxes = []
    result_masks = []
    result_polys = []

    bboxes_tif = []
    masks_tif = []
    polys_tif = []

    with open(os.path.join(output_dir, 'detection.pkl'), 'rb') as f:
        detections = pickle.load(f)

    for detection in detections:
        bboxes_tif.extend(detection['boxes'])
        masks_tif.extend(detection['masks'])
        polys_tif.extend(detection['polys'])

    # label, score, x1, y1, x2, y2
    bboxes_tif = np.array(bboxes_tif, dtype=np.float32)
    polys_tif = np.array(polys_tif, dtype=np.float32)

    temp_bboxes = torch.from_numpy(bboxes_tif).reshape(-1, 6)
    labels, scores, bboxes = torch.split(temp_bboxes, [1, 1, 4], dim=1)
    labels = labels.squeeze(-1)
    scores = scores.squeeze(-1)

    polys = torch.from_numpy(polys_tif).reshape(-1, 7)

    inds = remove_small_bboxes(bboxes, cfg.min_box_size)
    boxlist = BoxList(bboxes[inds])
    boxlist.add_field('labels', labels[inds])
    boxlist.add_field('scores', scores[inds])
    boxlist.add_field('polys', polys[inds])
    boxlist.add_field('mask_id', torch.arange(boxlist.bboxes.shape[0]))

    masks = [masks_tif[i] for i in inds]

    for x_start in range(0, width, nms_size - nms_overlap_size):
        for y_start in range(0, height, nms_size - nms_overlap_size):
            x_range = min(width - x_start, nms_size)
            y_range = min(height - y_start, nms_size)

            if x_range <= nms_overlap_size and x_start > 0:
                continue
            if y_range <= nms_overlap_size and y_start > 0:
                continue
            
            bboxes = boxlist.bboxes
            x_c = (bboxes[:, 0] + bboxes[:, 2]) / 2
            y_c = (bboxes[:, 1] + bboxes[:, 3]) / 2
            id_in_range = (x_c >= x_start) & (y_c >= y_start) & (x_c < x_start + x_range) & (y_c < y_start + y_range)
            id_out_range = ~id_in_range

            boxlist_in_range = boxlist[id_in_range]
            labels_in_range = boxlist_in_range.get_field('labels') 

            boxlist_in_range_cate = []
            for cate_id in range(len(CATEGORIES)):
                labels_per_cate = labels_in_range == cate_id
                bboxes_ = boxlist_in_range.bboxes[labels_per_cate]
                scores_ = boxlist_in_range.get_field('scores')[labels_per_cate]

                _, inds = nms_bbox(bboxes_, scores_, cfg.bbox_nms_threshold, gpu_id)

                boxlist_in_range_cate.append(boxlist_in_range[labels_per_cate][inds])

            boxlist_in_range = cat_boxlist(boxlist_in_range_cate)
            boxlist_out_range = boxlist[id_out_range]
            boxlist = cat_boxlist([boxlist_in_range, boxlist_out_range])
    
    bboxes_nms = boxlist.bboxes
    labels = boxlist.get_field('labels').tolist()
    scores = boxlist.get_field('scores').tolist()
    mask_id = boxlist.get_field('mask_id').tolist()
    masks_nms = [masks[id_][2:] for id_ in mask_id]

    for bbox, mask, label, score in zip(bboxes_nms, masks_nms, labels, scores):
        label = int(label)
        points = []
        points.append([CATEGORIES[label], score])
        bbox_ = np.array(bbox).reshape(-1, 2)
        points.extend([
            [bbox_[0, 0], bbox_[0, 1]],
            [bbox_[1, 0], bbox_[0, 1]],
            [bbox_[1, 0], bbox_[1, 1]],
            [bbox_[0, 0], bbox_[1, 1]]
        ])
        points.append(points[1])
        result_boxes.append(points)

        points = []
        points.append([CATEGORIES[label], score])
        mask_ = np.array(mask).reshape(-1, 2).tolist()
        points.extend(mask_)
        points.append(points[1])
        result_masks.append(points)
    
    # label, score, x, y, w, h, angle
    polys = boxlist.get_field('polys')
    inds = (polys[:, 4] > cfg.min_box_size) & (polys[:, 5] > cfg.min_box_size)
    polys = polys[inds]

    labels, scores, rboxes = torch.split(polys, [1, 1, 5], dim=1)
    labels = labels.squeeze(-1)
    scores = scores.squeeze(-1)
    boxlist = BoxList(rboxes)
    boxlist.add_field('labels', labels)
    boxlist.add_field('scores', scores)

    for x_start in range(0, width, nms_size - nms_overlap_size):
        for y_start in range(0, height, nms_size - nms_overlap_size):
            x_range = min(width - x_start, nms_size)
            y_range = min(height - y_start, nms_size)

            if x_range <= nms_overlap_size and x_start > 0:
                continue
            if y_range <= nms_overlap_size and y_start > 0:
                continue
            
            rboxes = boxlist.bboxes
            x_c = rboxes[:, 0]
            y_c = rboxes[:, 1]
           
            id_in_range = (x_c >= x_start) & (y_c >= y_start) & (x_c < x_start + x_range) & (y_c < y_start + y_range)
            id_out_range = ~id_in_range

            boxlist_in_range = boxlist[id_in_range]
            labels_in_range = boxlist_in_range.get_field('labels')

            boxlist_in_range_cate = []
            for cate_id in range(len(CATEGORIES)):
                labels_per_cate = labels_in_range == cate_id
                rboxes_ = boxlist_in_range.bboxes[labels_per_cate]
                scores_ = boxlist_in_range.get_field('scores')[labels_per_cate]
                _, inds = nms_rbox(rboxes_, scores_, cfg.rbox_nms_threshold, gpu_id)

                boxlist_in_range_cate.append(boxlist_in_range[labels_per_cate][inds])
            boxlist_in_range = cat_boxlist(boxlist_in_range_cate)
            boxlist = cat_boxlist([boxlist_in_range, boxlist[id_out_range]])

    labels = boxlist.get_field('labels')
    scores = boxlist.get_field('scores')
    rboxes = boxlist.bboxes

    pps = rboxes[:, -5:].contiguous()
    pp_iou = rbbox_iou(pps, pps, iou_type='iof')
    pp_iou[torch.arange(len(pp_iou)), torch.arange(len(pp_iou))] = 0
    if pp_iou.numel():
        max_iou, ids = pp_iou.max(dim=-1)
        pp_inds = max_iou < 0.7
    else:
        pp_inds = torch.zeros((0, ), dtype=torch.bool)
    rboxes = rboxes[pp_inds]
    labels = labels[pp_inds].tolist()
    scores = scores[pp_inds].tolist()

    for rbox, label, score in zip(rboxes, labels, scores):
        label = int(label)
        points = []
        rbox = rbox.tolist()
        points.append([CATEGORIES[label], score])
        box = cv2.boxPoints(((rbox[0], rbox[1]), (rbox[2], rbox[3]), rbox[4]))
        points.extend(box.tolist())
        points.append(points[1])
        result_polys.append(points)

    save(output_dir, result_boxes, result_polys, result_masks)
    
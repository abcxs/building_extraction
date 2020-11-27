import config as cfg
import cv2
import gdal
import itertools
import mmcv
import numpy as np
import os
import pickle
import torch
from approx_poly import approx
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from tqdm import tqdm
from utils import check_image, cut_into_blocks, findContours, logger, nms_iof

from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def inference_detector(model, img, test_pipeline):
    device = next(model.parameters()).device  # model device
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
   
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]
    return result

def init_detector(config, checkpoint=None, device_id=0):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
    fuse_conv_bn(model)
    model = MMDataParallel(model, device_ids=[device_id])
    model.cfg = config  # save the config in the model for convenience
    model.eval()
    return model

def parse_result(result, score_thr, lt, approx_polygon=False):
    bbox_result, segm_result = result

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    inds = np.where(bboxes[:, -1] > score_thr)[0]

    _, inds_iof = nms_iof(bboxes[inds], cfg.iof_thr)
    inds = inds[inds_iof]

    segms = list(itertools.chain(*segm_result))
    segms = np.array(segms).astype(np.uint8)

    bboxes_img = []
    masks_img = []
    polys_img = []
    
    lt = np.array(lt)
    for i in inds:
        bbox = bboxes[i][:-1]
        score = bboxes[i][-1]
        label = labels[i]
        mask = segms[i]
        
        points = [label, score]
        bbox_ = bbox.reshape(-1, 2) + lt
        points.extend(bbox_.reshape(-1).tolist())
        bboxes_img.append(points)

        contours, _ = findContours(mask[..., None], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 0:
            bboxes_img.pop()
            continue
        contours = np.concatenate(contours, axis=0)
        contour = cv2.convexHull(contours)

        points = [label, score]
        if approx_polygon:
            contour_ = approx(contour.copy()).reshape(-1, 2) + lt
        else:
            contour_ = contour.reshape(-1, 2) + lt
        points.extend(contour_.reshape(-1).tolist())
        masks_img.append(points)

        rect = cv2.minAreaRect(contour)
        x, y = rect[0]
        w, h = rect[1]
        a = rect[2]
        points = [label, score, x + lt[0], y + lt[1], w, h, a]
        polys_img.append(points)

    result = dict(boxes=bboxes_img, masks=masks_img, polys=polys_img) 
    return result

def detect(process_id, gpu_id, config_file, checkpoint, tif_file, piece_list, output_dir, approx_polygon=False):
    if check_image(tif_file):
        ds = cv2.imread(tif_file)
    else:
        ds = gdal.Open(tif_file, gdal.GA_ReadOnly)

    model = init_detector(config_file, checkpoint, device_id=gpu_id)
    cfg_ = model.cfg.copy()
    cfg_.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    test_pipeline = Compose(cfg_.data.test.pipeline)

    detections = []
    for piece in tqdm(piece_list):
        lt = piece[:2]
        image, image_block_list = cut_into_blocks(ds, piece, cfg)
        if image is None:
            continue
        
        for image_block in image_block_list:
            x, y, w, h = image_block
            image_patch = image[y: y + h, x: x + w].copy()
            if (image_patch == 0).all() or (image_patch == 255).all():
                continue
            
            result = inference_detector(model, image_patch, test_pipeline)

            detection = parse_result(result, cfg.score_thr, [lt[0] + x, lt[1] + y], approx_polygon)
            detections.append(detection)

    with open(os.path.join(output_dir, 'detection_%d.pkl' % process_id), 'wb') as f:
        pickle.dump(detections, f)

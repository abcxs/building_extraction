# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import gdal
import logging
import numpy as np
import os
import time
import torch
from collections import OrderedDict
from contextlib import contextmanager
from mmcv.ops import bbox_overlaps
from mmcv.ops.nms import nms
from PIL import Image

from mmdet.rotation_libs.rotate_polygon_nms import rotate_gpu_nms


def cut_into_(width, height, size, overlap_size):
    patch_list = []
    for y in range(0, height, size - overlap_size):
        for x in range(0, width, size - overlap_size):

            patch_height = min(size, height - y)
            patch_width = min(size, width - x)

            if patch_width <= overlap_size and x > 0:
                continue
            if patch_height <= overlap_size and y > 0:
                continue

            start_y = y + patch_height - size
            start_x = x + patch_width - size
            patch_height = size
            patch_width = size

            if start_y < 0:
                start_y = 0
                patch_height = height

            if start_x < 0:
                start_x = 0
                patch_width = width

            patch_list.append([start_x, start_y, patch_width, patch_height])
    return patch_list

def cut_into_pieces(tif_file, cfg):
    piece_size = cfg.piece_size
    piece_overlap_size = cfg.piece_overlap_size

    if check_image(tif_file):
        ds = Image.open(tif_file)
        width, height = ds.size
    else:
        try:
            ds = gdal.Open(tif_file, gdal.GA_ReadOnly)
            width = ds.RasterXSize
            height = ds.RasterYSize
        except:
            return []

    if piece_size <= 0:
        return [[0, 0, width, height]]

    piece_list = cut_into_(width, height, piece_size, piece_overlap_size)
    return piece_list


def cut_into_blocks(ds, piece, cfg):
    detect_size = cfg.detect_size
    detect_overlap_size = cfg.detect_overlap_size

    lt_x, lt_y, width, height = piece

    if isinstance(ds, np.ndarray):
        image = ds[lt_y: lt_y + height, lt_x: lt_x + width]
    else:
        try:
            image = ds.ReadAsArray(lt_x, lt_y, width, height)
        except:
            return None, []

        if (image == 255).all() or (image == 0).all():
            return None, []

        image = image.astype(np.uint8).transpose([1, 2, 0])[:, :, ::-1]

    block_list = cut_into_(width, height, detect_size, detect_overlap_size)
    
    return image, block_list


def get_logger():
    logger = logging.getLogger('building extraction')
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


def nms_rbox(rboxes, scores, nms_thresh, gpu_id=0):
    rboxes = rboxes.numpy()
    scores = scores.numpy()
    dets = np.concatenate((rboxes, scores[:, None]), axis=1)
    if nms_thresh <= 0 or len(dets) <= 0:
        return dets, np.arange(len(dets))
    inds = rotate_gpu_nms(dets, float(nms_thresh), gpu_id)
    dets = dets[inds]
    return dets, inds


def nms_bbox(bboxes, scores, nms_thresh, gpu_id=0):
    bboxes = bboxes.to('cuda:%d' % gpu_id)
    scores = scores.to('cuda:%d' % gpu_id)
    dets, inds = nms(bboxes, scores, nms_thresh)
    return dets, inds


def remove_small_bboxes(bboxes, min_size):
    assert isinstance(bboxes, torch.Tensor)
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    inds = (w >= min_size) & (h >= min_size)
    return torch.nonzero(inds, as_tuple=False).squeeze(1)

def get_gpus():
    gpus = os.environ.get('BUILDING_CUDA', None)
    num = torch.cuda.device_count()
    if gpus:
        gpus = list(map(int, gpus.replace(' ', '').split(',')))
        gpus = [id_ for id_ in gpus if id_ < num]
    if not gpus:
        gpus = list(range(num))
    return gpus

def remove_inside_boxes(dets):
    if len(dets) <= 0:
        return dets, np.array([], dtype=np.int64)
    # N, N, 5
    diff = dets[:, None] - dets[None]
    diff = diff[..., :4]
    diff[..., 2:] *= -1
    diff = np.all(diff > 0, axis=-1)
    diff = np.sum(diff, axis=-1)
    inds = np.where(diff == 0)[0]
    dets = dets[inds]
    return dets, inds


def nms_iof(dets, iou_thresh):
    if len(dets) <= 0:
        return dets, np.array([], dtype=np.int64)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    ids = scores.argsort()[::-1]
    keep = []
    while len(ids) > 0:
        i = ids[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[ids])
        yy1 = np.maximum(y1[i], y1[ids])
        xx2 = np.minimum(x2[i], x2[ids])
        yy2 = np.minimum(y2[i], y2[ids])
        w = np.maximum(xx2 - xx1 + 1, 0)
        h = np.maximum(yy2 - yy1 + 1, 0)
        ovr = w * h / area[ids]
        ids = ids[ovr < iou_thresh]
    keep = np.array(keep, dtype=np.int64)
    
    return dets[keep], keep

def check_image(path):
    ext = os.path.splitext(path)[1]
    if ext in ['.png', '.jpg', '.JPEG']:
        return True
    return False

def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton

class TimeBuffer(object):
    def __init__(self):
        self.time_buffer = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.time_buffer:
                self.time_buffer[key] = []
            self.time_buffer[key].append(var)

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.time_buffer:
            values = np.array(self.time_buffer[key][-n:])
            avg = np.sum(values) / len(values)
            self.output[key] = avg
        self.ready = True

class TimeRecord(object):
    def __init__(self, time_key):
        self.start_time = time.time()
        self.end_time = -1
        self.time_elapse = -1
        self.time_key = time_key
        logger.info("{0} starts".format(self.time_key))

    def stop(self):
        self.end_time = time.time()
        self.time_elapse = self.end_time - self.start_time

        logger.info("{0} ends, time elapse: {1}".format(self.time_key, self.time_elapse))


@contextmanager
def time_record(time_key):
    record = TimeRecord(time_key)
    yield record
    record.stop()


def enable_time_record(func, time_key=None):
    time_key = time_key if time_key else func.__name__

    def wrapper(*args, **kwargs):
        record = TimeRecord(time_key)
        func_ret = func(*args, **kwargs)
        record.stop()
        return func_ret
    return wrapper

class TestCase(object):
    pass


def test_rbox():
    rboxes = np.array([[50, 50, 100, 100, 0],
                       [60, 60, 100, 100, 0],
                       [50, 50, 100, 100, -45.],
                       [200, 200, 100, 100, 0.],
                       [0, 0, 100, 100, 30],
                       [0, 0, 100, 100, 28]], dtype=np.float32)

    scores = np.array([0.99, 0.88, 0.66, 0.77, 0.4, 0.3], dtype=np.float32)
    dets, inds = nms_rbox(rboxes, scores, 0.3)
    print(dets)
    print(inds, inds.dtype)


def test_bbox():
    bboxes = np.array([[49.1, 32.4, 51.0, 35.9],
                       [49.3, 32.9, 51.0, 35.3],
                       [49.2, 31.8, 51.0, 35.4],
                       [35.1, 11.5, 39.1, 15.7],
                       [35.6, 11.8, 39.3, 14.2],
                       [35.3, 11.5, 39.9, 14.5],
                       [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
    scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],
                      dtype=np.float32)
    iou_threshold = 0.6
    dets, inds = nms_bbox(bboxes, scores, iou_threshold)
    print(dets)
    print(inds, inds.dtype)

def test_nms_iof():
    # bboxes = np.array([[49.1, 32.4, 51.0, 35.9],
    #                    [49.3, 32.9, 51.0, 35.3],
    #                    [49.2, 31.8, 51.0, 35.4],
    #                    [35.1, 11.5, 39.1, 15.7],
    #                    [35.6, 11.8, 39.3, 14.2],
    #                    [35.3, 11.5, 39.9, 14.5],
    #                    [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
    # scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],
    #                   dtype=np.float32)
    # dets = np.concatenate([bboxes, scores[:, None]], axis=-1)
    dets = np.zeros((0, 5))
    iou_threshold = 0.6
    dets, inds = nms_iof(dets, iou_threshold)
    print(dets)
    print(inds, inds.dtype)

def test_remove_inside_boxes():
    bboxes = np.array([[49.1, 32.4, 51.0, 35.9],
                       [49.3, 32.9, 51.0, 35.3],
                       [49.2, 31.8, 51.0, 35.4],
                       [35.1, 11.5, 39.1, 15.7],
                       [35.6, 11.8, 39.3, 14.2],
                       [35.3, 11.5, 39.9, 14.5],
                       [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
    scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],
                      dtype=np.float32)
    dets = np.concatenate([bboxes, scores[:, None]], axis=-1)
    # dets = np.zeros((0, 5))
    dets, inds = remove_inside_boxes(dets)
    print(dets)
    print(inds, inds.dtype)


def test_remove_small_bboxes():
    bboxes = np.array([[0, 0, 100, 100],
                       [10, 10, 20, 20]], dtype=np.float32)
    min_size = 12
    inds = remove_small_bboxes(bboxes, min_size)
    print(inds)


def test_cut_into_pieces():
    import config as cfg

    ds = TestCase()
    ds.RasterXSize = 5860
    ds.RasterYSize = 886
    print(cut_into_pieces(ds, cfg))

def test_findCountours():
    import numpy as np
    from approx_poly import approx
    mask = np.zeros([100, 100], dtype=np.uint8)
    mask[20:40, 30:60] = 1
    contours, _ = findContours(mask[..., None], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), contours[0].shape)
    contours = np.concatenate(contours, axis=0)
    contour = cv2.convexHull(contours)
    print(contour.shape)
    contour = approx(contour)
    print(contour.shape)

if __name__ == '__main__':
    test_remove_inside_boxes()

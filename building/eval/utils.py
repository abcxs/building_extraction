import numpy as np


def geo2cr(geoTransform, x, y):
    dTemp = geoTransform[1] * geoTransform[5] - geoTransform[2] * geoTransform[4]
    dcol = (geoTransform[5] * (x - geoTransform[0]) - geoTransform[2] * (
            y - geoTransform[3])) / dTemp + 0.5
    drow = (geoTransform[1] * (y - geoTransform[3]) - geoTransform[4] * (
            x - geoTransform[0])) / dTemp + 0.5
    return dcol, drow

def cr2geo(geoTransform, x, y):
    xp = geoTransform[0] + x * geoTransform[1] + y * geoTransform[2]
    yp = geoTransform[3] + x * geoTransform[4] + y * geoTransform[5]
    return xp, yp

def area_of_boxes(box1, box2):
    '''
    :param box1: target box, [N, 4]
    :param box2: predict box, [M, 4]
    :return:
        area  [N, M]
    '''
    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = np.clip((rb - lt), a_min=0, a_max=None)
    inter_area = wh[:, :, 0] * wh[:, :, 1]

    return inter_area

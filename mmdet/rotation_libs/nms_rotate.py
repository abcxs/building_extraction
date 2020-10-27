import numpy as np
import cv2
from rotation.rotation_libs.rotate_polygon_nms import rotate_gpu_nms
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def rotate_cpu_nms(dets, threshold):
    '''

    :param dets: [N, 6] x_ctr, y_ctr, width, height, angle, score
    :param threshold: float
    :return:
    '''
    scores = dets[:, -1]

    order = scores.argsort()[::-1]
    ndets = dets.shape[0]

    suppressed = np.zeros((ndets), dtype=np.int)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((dets[i, 0], dets[i, 1]), (dets[i, 2], dets[i, 3]), dets[i, 4])

        area_r1 = dets[i, 2] * dets[i, 3]
        for _j in range(_i + 1, ndets):
            # tic = time.time()
            j = order[_j]
            if suppressed[j] == 1:
                continue
            r2 = ((dets[j, 0], dets[j, 1]), (dets[j, 2], dets[j, 3]), dets[j, 4])
            area_r2 = dets[j, 2] * dets[j, 3]
            ovr = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                ovr = int_area * 1.0 / (area_r1 + area_r2 - int_area)
                print(_i, _j, ovr)

            if ovr >= threshold:
                suppressed[j] = 1
    return keep


if __name__ == "__main__":
    boxes = np.array([
        [0, 0, 80, 80, 0, 0.99],
        [20, 20, 100, 40, 45, 0.88],  # keep 0.68
        [20, 20, 100, 40, -45, 0.66],  # discard 0.70
    ])

    thresh = np.float(0.3)
    import matplotlib.pyplot as plt
    import shapely.geometry as sg

    fig, axs = plt.subplots()
    for box in boxes:
        points = cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), box[4]))
        poly = sg.Polygon(points)
        xs, ys = poly.exterior.xy
        axs.fill(xs, ys, alpha=.25, fc='r', ec='none')
    plt.show()
    keep = rotate_cpu_nms(boxes, thresh)
    print(keep)
    print(rotate_gpu_nms(boxes.astype(np.float32), 0.3))

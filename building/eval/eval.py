import config as cfg
import matplotlib.pyplot as plt
import numpy as np
import os
import xml.etree.ElementTree as ET


def parse_gt(filename):
    objects = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        splitlines = [x.strip().split(' ')  for x in lines]
        for splitline in splitlines:
            object_struct = {}

            object_struct['name'] = 'building'

            points = [float(p) for p in splitline]
            x_min = min(points[0::2])
            x_max = max(points[0::2])
            y_min = min(points[1::2])
            y_max = max(points[1::2])
            object_struct['bbox'] = [x_min, y_min, x_max, y_max]

            w = x_max - x_min
            h = y_max - y_min
            object_struct['area'] = w * h

            object_struct['difficult'] = 0
            # min_area = 15 * 15
            # if object_struct['area'] < min_area:
            #     object_struct['difficult'] = 1

            objects.append(object_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip().split(' ')[0] for x in lines]
    sites = [x.strip().split(' ')[1] for x in lines]
    recs = []
    precs = []
    for imagename, site in zip(imagenames, sites):
        npos = 0
        gt = parse_gt(annopath.format(imagename))
        R = [obj for obj in gt if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)

        det = [False] * len(R)
        npos = npos + sum(~difficult)
        gt = {'bbox': bbox, 'difficult': difficult, 'det': det}

        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]

        confidence = np.array([float(x[1]) for x in splitlines if x[0] == imagename])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines if x[0] == imagename])
        BB = BB.reshape(-1, 4, 2)
        xymin = BB.min(axis=1)
        xymax = BB.max(axis=1)
        BB = np.concatenate([xymin, xymax], axis=-1)

        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        
        nd = BB.shape[0]
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = gt['bbox'].astype(float)

            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + 
                        (BBGT[:, 2] - BBGT[:, 0] + 1.) * 
                        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                lt_id = np.nonzero(ovmax > ovthresh)[0]
            if ovmax > ovthresh:
                for t in lt_id:
                    gt['det'][t] = 1
                tp[d] = 1
                gt['det'][jmax] = 1
            else:
                fp[d] = 1.
        rec = sum(gt['det']) / npos
        prec = sum(tp) / nd
        f1 = 2 * prec * rec / (prec + rec)
        print('gt:', npos, 'detect num:', sum(gt['det']), 'recall:', rec ,'prec:', prec, 'f1:', f1, 'site:', site)
        
        recs.append(rec)
        precs.append(prec)

    return sum(recs) / len(recs), sum(precs) / len(precs), 0

def main():
    detpath = os.path.join(cfg.prediction_dir, '%s.txt' % cfg.eval_type)
    annopath = os.path.join(cfg.gt_dir, '{:s}.txt')
    imagesetfile = os.path.join(cfg.base_dir, 'eval/imageset.txt')
    ovthresh = cfg.ovthresh

    classnames = ['building']


    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=ovthresh,
             use_07_metric=True)
        print('recal:',rec, 'prec:', prec, 'ap:', ap)


if __name__ == '__main__':
    main()

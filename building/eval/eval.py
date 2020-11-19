import cfg
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

            object_struct['name'] = splitline[0]

            points = [float(p) for p in splitline[1:]]
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

def voc_eval1(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    recs = []
    precs = []
    for imagename in imagenames:
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
                if not gt['difficult'][jmax]:
                    tp[d] = 1
                    gt['det'][jmax] = 1
            else:
                fp[d] = 1.
        rec = sum(gt['det']) / npos
        prec = sum(tp) / nd
        print('gt:', npos, 'detect num:', sum(gt['det']), 'recall:', rec ,'prec:', prec, 'site:', imagename)
        
        recs.append(rec)
        precs.append(prec)

    return sum(recs) / len(recs), sum(precs) / len(precs), 0

def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_gt(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)

        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_names = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)
    BB = BB[sorted_ind, :]
    image_names = [image_names[x] for x in sorted_ind]

    nd = len(image_names)
    tp_overlap = np.zeros(nd)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    for d in range(nd):
        R = class_recs[image_names[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

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
            ## if there exist 2
            jmax = np.argmax(overlaps)
            lt_id = np.nonzero(ovmax > ovthresh)[0]
            for id_ in lt_id:
                R['det'][id_] = 1

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    tp_overlap[d] = 1.
                    R['det'][jmax] = 1
                else:
                    tp_overlap[d] = 1.
                    # change this
                    # fp[d] = 1.
        else:
            fp[d] = 1.

    print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    tp_overlap = np.cumsum(tp_overlap)

    rec = tp / float(npos)
    print(tp_overlap[-1] / nd)
    prec = tp_overlap / np.maximum(tp_overlap + fp, np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec, use_07_metric)

    t1 = 0
    for imagename in imagenames:
        t1 += sum(class_recs[imagename]['det'])
    print(t1 / npos)

    return rec, prec, ap

def main():
    detpath = os.path.join(cfg.input_dir, 'result/%s/{:s}.txt' % cfg.detect_type)
    annopath = os.path.join(cfg.input_dir, 'data/{:s}.txt')
    imagesetfile = os.path.join(cfg.input_dir, 'imageset.txt')
    ovthresh = cfg.ovthresh

    classnames = ['building']

    classaps = []
    map = 0

    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval1(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=ovthresh,
             use_07_metric=True)
        map = map + ap
        print('recal:',rec, 'prec:', prec, 'ap:', ap)
        classaps.append(ap)

    map = map / len(classnames)
    print('map:', map)
    classaps = 100 * np.array(classaps)
    print('classaps: ', classaps)

    
if __name__ == '__main__':
    main()

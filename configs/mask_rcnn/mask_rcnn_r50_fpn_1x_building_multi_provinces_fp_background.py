_base_ = './mask_rcnn_r50_fpn_1x_building_multi_provinces_fp.py'
data = dict(
    train=dict(
        filter_empty_gt=False))

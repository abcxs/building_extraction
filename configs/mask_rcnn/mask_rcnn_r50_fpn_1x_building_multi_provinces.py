_base_ = './mask_rcnn_r50_fpn_1x_building.py'
data_root = 'data/building/multi_provinces/'
data = dict(
    train=dict(
        ann_file=data_root + 'train/train.json',
        img_prefix=data_root + 'train/JPEGImages/'),
    val=dict(
        ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/JPEGImages/'),
    test=dict(
        ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/JPEGImages/'))
        
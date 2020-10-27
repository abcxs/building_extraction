_base_ = './mask_rcnn_r50_fpn_1x_building.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 704), (1333, 768), (1333, 832), (1333, 896),
                   (1333, 960), (1333, 1024)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data_root='data/building/'
sites = ['fujian', 'gansu', 'guangdong', 'guizhou', 'hainan', 'hubei', 'hunan', 'jiangxi', 'multi_provinces', 'qinghai', 'sichuan', 'xizang', 'yunnan', 'zhejiang']
train_ann_files = [data_root + '%s/train/train.json' % site for site in sites]
train_img_prefixs = [data_root + '%s/train/JPEGImages/' % site for site in sites]
val_ann_files = [data_root + '%s/val/val.json' % site for site in sites]
val_img_prefixs = [data_root + '%s/val/JPEGImages/' % site for site in sites]
test_ann_files = val_ann_files
test_img_prefixs = val_img_prefixs
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        ann_file=train_ann_files,
        img_prefix=train_img_prefixs,
        pipeline=train_pipeline,
        filter_empty_gt=False),
    val=dict(
        ann_file=val_ann_files,
        img_prefix=val_img_prefixs,),
    test=dict(
        ann_file=test_ann_files,
        img_prefix=test_img_prefixs))
optimizer = dict(lr=0.01)
fp16 = dict(loss_scale=512.)

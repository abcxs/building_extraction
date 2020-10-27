_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/building_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
optimizer = dict(lr=0.02)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            gpu_assign_thr=1024)))

test_cfg = dict(
    rcnn=dict(
        max_per_img=1000))

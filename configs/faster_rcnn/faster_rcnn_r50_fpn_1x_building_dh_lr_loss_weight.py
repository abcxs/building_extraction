_base_ = './faster_rcnn_r50_fpn_1x_building_dh_lr_loss.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(loss_weight=1.0),
            loss_bbox=dict(loss_weight=1.0))))
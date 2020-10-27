_base_ = './faster_rcnn_r50_fpn_1x_building_dh_lr_loss.py'
model = dict(
    roi_head=dict(reg_roi_scale_factor=1))

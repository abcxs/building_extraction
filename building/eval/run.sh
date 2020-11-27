BASE_DIR=/home/ndrcchkygb/code/mmdetection
CFG=$BASE_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_building_base_fp_background_ms.py
MODEL=$BASE_DIR/work_dirs/mask_rcnn_r50_fpn_1x_building_base_fp_background_ms/epoch_12.pth
SITE=qinghai
INPUT_DIR=/home/ndrcchkygb/data/building/dst/$SITE/eval/gt
OUTPUT_DIR=/home/ndrcchkygb/data/building/dst/$SITE/eval/prediction
python ../tif_process/main.py $INPUT_DIR $OUTPUT_DIR $CFG $MODEL --eval
# OUTPUT_DIR=/home/ndrcchkygb/data/building/dst/$SITE/eval/visual
# python ../tif_process/main.py $INPUT_DIR $OUTPUT_DIR $CFG $MODEL
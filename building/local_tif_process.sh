INPUT_DIR=/home/ndrcchkygb/data/test_data/input
OUTPUT_DIR=/home/ndrcchkygb/data/test_data/output
CFG_DIR=/home/ndrcchkygb/project/cfg_mmd
MODEL_DIR=/home/ndrcchkygb/project/model_mmd
export BUILDING_CUDA=0,1,2,3,4,5,6,7
export BASE_INPUT_DIR=$INPUT_DIR
model=base
input=舟曲县立节镇北山村滑坡
/home/ndrcchkygb/anaconda3/envs/open-mmlab/bin/python tif_process/main.py $INPUT_DIR/$input $OUTPUT_DIR $CFG_DIR/building.py $MODEL_DIR/$model.pth
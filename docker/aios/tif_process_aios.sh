OUTPUT_DIR=${output_dir:/private/building/output}
CFG_DIR=/private/building/cfg
MODEL_DIR=/private/building/model
config=${config:building}
PY_ARGS=${@:1}
cd /root/mmdetection
python building/tif_process/main.py $data_dir $OUTPUT_DIR $CFG_DIR/$config.py $MODEL_DIR/$model.pth $PY_ARGS

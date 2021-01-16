INPUT_DIR=/private/building/data
OUTPUT_DIR=/private/building/result
CFG_DIR=/private/building/cfg
MODEL_DIR=/private/building/model
echo '输入目录:'
read input
echo -n '输入模型(  '
cd $MODEL_DIR
for file in `ls *.pth`
do
	echo -n ${file%%.*}
	echo -n '  '
done
echo ')\n'
read model
cd /root/mmdetection
python building/tif_process/main.py $INPUT_DIR/$input $OUTPUT_DIR $CFG_DIR/building.py $MODEL_DIR/$model.pth

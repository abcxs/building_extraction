INPUT_DIR=/private/data
OUTPUT_DIR=/private/result
CFG_DIR=/private/cfg
MODEL_DIR=/private/model
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
python tif_process/main.py $INPUT_DIR/$input $OUTPUT_DIR $CFG_DIR/building.py $MODEL_DIR/$model.pth

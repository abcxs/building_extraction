echo '处理/home/ndrcchkygb/project/data下软连接目录下的所有tif文件，并输出至/home/ndrcchkygb/project/result目录下相对路径'

INPUT_DIR=/home/ndrcchkygb/project/data
OUTPUT_DIR=/home/ndrcchkygb/project/result
CFG_DIR=/home/ndrcchkygb/project/cfg_mmd
MODEL_DIR=/home/ndrcchkygb/project/model_mmd
export BUILDING_CUDA=0,1,2,3,4,5,6,7
export BASE_INPUT_DIR=$INPUT_DIR
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
echo '是否拟合（会消耗更多时间y/n）'
read approx
cd /home/ndrcchkygb/code/mmdetection/building
case $approx in
    [yY][eE][sS]|[yY])
		/home/ndrcchkygb/anaconda3/envs/open-mmlab/bin/python tif_process/main.py $INPUT_DIR/$input $OUTPUT_DIR $CFG_DIR/building.py $MODEL_DIR/$model.pth --approx_polygon
		;;

    [nN][oO]|[nN])
		/home/ndrcchkygb/anaconda3/envs/open-mmlab/bin/python tif_process/main.py $INPUT_DIR/$input $OUTPUT_DIR $CFG_DIR/building.py $MODEL_DIR/$model.pth
       	;;

    *)
		echo "Invalid input..."
		exit 1
		;;
esac
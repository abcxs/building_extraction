BASE=/home/ndrcchkygb/project
docker run -it --rm \ 
-v $BASE/cfg_mmd:/private/cfg -v \
-v $BASE/model_mmd:/private/model \
-v $BASE/data:/private/data \
-v $BASE/result:/private/result \
--gpus all building_extraction:1.0
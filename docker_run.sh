DATASET_PATH="/media"
docker run --gpus=all -v ${PWD}:/dy -v $DATASET_PATH:/datasets  -d  -it --name dy_seg_tf2 -p 15016:15016 seg_tf2:1


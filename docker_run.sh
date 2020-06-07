docker run --runtime=nvidia -v ${PWD}:/dy -v $DATASET_PATH:/datasets  -d  -it --name seg_tf2 -p 15011:15011 seg_tf2:1

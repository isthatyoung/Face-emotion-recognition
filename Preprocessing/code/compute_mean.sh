DIR=/home/ubuntu/Face-emotion-recognition-master/Preprocessing/
/home/ubuntu/caffe/build/tools/compute_image_mean $DIR/data/img_train_lmdb/ $DIR/data/train_mean.binaryproto
/home/ubuntu/caffe/build/tools/compute_image_mean $DIR/data/img_val_lmdb/ $DIR/data/val_mean.binaryproto
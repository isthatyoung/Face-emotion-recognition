#!/usr/bin/env sh
MY=/home/ubuntu/Face-emotion-recognition-master/Preprocessing/data

echo "Create train lmdb.."
rm -rf $MY/img_train_lmdb
/home/ubuntu/caffe/build/tools/convert_imageset \
--shuffle \
--resize_height=48 \
--resize_width=48 \
--gray \
/home/ubuntu/Face-emotion-recognition-master/Preprocessing/data/face-alignment-fer2013/train \
$MY/train_list.txt \
$MY/img_train_lmdb

echo "Create test lmdb.."
rm -rf $MY/img_val_lmdb
/home/ubuntu/caffe/build/tools/convert_imageset \
--shuffle \
--resize_width=48 \
--resize_height=48 \
--gray \
/home/ubuntu/Face-emotion-recognition-master/Preprocessing/data/face-alignment-fer2013/val \
$MY/validation_list.txt \
$MY/img_val_lmdb

echo "All Done.."

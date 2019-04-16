#!/usr/bin/env sh
MY=/home/ubuntu/Face-emotion-recognition-master/Preprocessing/data

#echo "Create CK48 train lmdb.."
#rm -rf $MY/CK_train_lmdb
#/home/ubuntu/caffe/build/tools/convert_imageset \
#--shuffle \
#--resize_height=48 \
#--resize_width=48 \
#--gray \
#/home/ubuntu/Face-emotion-recognition-master/Preprocessing/data/face-alignment-cohn-kanade-images \
#$MY/CK_train_list.txt \
#$MY/CK_train_lmdb

echo "Create CK48 test lmdb.."
rm -rf $MY/CK_test_lmdb
/home/ubuntu/caffe/build/tools/convert_imageset \
--shuffle \
--resize_height=48 \
--resize_width=48 \
--gray \
/home/ubuntu/Face-emotion-recognition-master/Preprocessing/data/face-alignment-cohn-kanade-images \
$MY/CK_test_list.txt \
$MY/CK_test_lmdb
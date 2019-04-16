#!/usr/bin/env bash
DIR=/home/ubuntu/Face-emotion-recognition-master
python ./confusionMatrix_Recall_Precision_F1Scroe_Caffe.py --proto $DIR/Test/code/FaceEmotionNet_deploy.prototxt --model $DIR/Train/data/FaceEmotionNet.caffemodel --mean $DIR/Preprocessing/data/test_mean.binaryproto --db_type lmdb --db_path $DIR/Preprocessing/data/img_test_lmdb
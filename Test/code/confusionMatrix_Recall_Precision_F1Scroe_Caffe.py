#!/usr/bin/python

# Author: SeyyedHossein Hasanpour copyright 2017, license GPLv3.
# Seyyed Hossein Hasan Pour:
# Coderx7@Gmail.com
# Changelog:
# 2015:
# initial code to calculate confusionmatrix by Axel Angel
# 7/3/2016:(adding new features-by-hossein) 
# added mean subtraction so that, the accuracy can be reported accurately just like caffe when doing a mean subtraction
# 01/03/2017:
# removed old codes and Added Recall/Precision/F1-Score as well 
# 03/05/2017
# Added ConfusionMatrix which was mistakenly ommited before.
#info:
#python confusionMatrix_convnet_test.py --proto cifar10_deploy_94_68.prototxt --model cifar10_deploy_94_68.caffemodel --mean mean.binaryproto --db_type lmdb --db_path cifar10_test_lmdb  



import sys
import caffe
import numpy as np
import lmdb
import argparse
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import matplotlib
matplotlib.use('TkAgg')

def flat_shape(x):
    "Returns x without singleton dimension, eg: (1,28,28) -> (28,28)"
    return x.reshape(filter(lambda s: s > 1, x.shape))

def db_reader(fpath, type='lmdb'):
    if type == 'lmdb':
	    return lmdb_reader(fpath)
    else:
	   return leveldb_reader(fpath)


def plot_confusion_matrix(cm #confusion matrix
                         ,classes 
                          ,normalize=False
                          ,title='Confusion matrix'
                          ,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("confusion matrix is normalized!")


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



	   
def lmdb_reader(fpath):
    import lmdb
    lmdb_env = lmdb.open(fpath)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum).astype(np.uint8)
        yield (key, flat_shape(image), label)

def leveldb_reader(fpath):
    import leveldb
    db = leveldb.LevelDB(fpath)

    for key, value in db.RangeIter():
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum).astype(np.uint8)
        yield (key, flat_shape(image), label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', help='path to the network prototxt file(deploy)', type=str, required=True)
    parser.add_argument('--model', help='path to your caffemodel file', type=str, required=True)
    parser.add_argument('--mean', help='path to the mean file(.binaryproto)', type=str, required=True)
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--db_type', help='lmdb or leveldb', type=str, required=True)
    parser.add_argument('--db_path', help='path to your lmdb/leveldb dataset', type=str, required=True)
    args = parser.parse_args()

   # Extract mean from the mean image file
    mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
    f = open(args.mean, 'rb')
    mean_blobproto_new.ParseFromString(f.read())
    mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    f.close()
	
   # CNN reconstruction and loading the trained weights	
    net = caffe.Net(args.proto, args.model, caffe.TEST)
   # You may also use set_mode_cpu() if you didnt compile caffe with gpu support
    caffe.set_mode_gpu() 
	
    print ("args", vars(args))
	
    reader = db_reader(args.db_path, args.db_type.lower())
    
    predicted_lables=[]
    true_labels = []
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    for i, image, label in reader:
        image_caffe = image.reshape(1, *image.shape)
        #print 'image shape: ',image_caffe.shape
        out = net.forward_all(data=np.asarray([ image_caffe ])- mean_image)
        plabel = int(out['prob'][0].argmax(axis=0))
        predicted_lables.append(plabel)	
        true_labels.append(label)
        print(i,' processed!')
	
    print( classification_report(y_true=true_labels,
	                             y_pred=predicted_lables,
	                             target_names=class_names))
    cm = confusion_matrix(y_true=true_labels,
	                        y_pred=predicted_lables)	
    print(cm)							
    # Compute confusion matrix
    cnf_matrix = cm
    np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
						  title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
						  title='Normalized confusion matrix')

    plt.show()
							

# Face-emotion-recognition
An implementation of face emotion recognition by Convolutional neurla network.

## Requirements
* Caffe 1.0.0
* Python 2.7/3.5
* OpenCV (cv2)
* Numpy
* [mtcnn](https://github.com/ipazc/mtcnn)

## Description
As for humans, we classify emotions all the time without knowing it. 
We can see if someone is happy or sad or frustrated and in need of help. 
However, it is a very complex problem that involves many subtleties about facial expression. 
Even just the tiniest change in someone’s face can be a signal of a different emotion. 
Training models that understand human emotions will be critical to build truly intelligent machine that can interact with us likes humans do. 

In this project, we build a classifier to recognize basic human emotion from facial expression by two steps: face detection and CNN based recognition. 

## Datasets
### Training
**FER2013 database** from Kaggle
[Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
 
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/images/Figure%201.png" width = "359" height = "169" align=center />
</div>  
This dataset has 7 facial expression categories (angry, disgust, fear, happy, sad, surprise and neutral).<br> 
28709 training set + 3589 validation set of 48px * 48px grayscale images.  

### Testing
* 3589 Testing set of FER2013.

## Main tasks
* Preprocessing (recent accomplished)
1. Extract and create images for FER2013 from csv
2. Face detection
3. Face alignment
4. Convert to LMDB format
5. Compute mean

* Training & Testing (recent accomplished)
1. Build model
2. Adjust hyper-parameters
3. Train model

* Predicting (recent accomplished)

## Preprocessing
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/images/Figure%202.png" width = "311" height = "125" align=center />
</div>  

## Training & Testing
Totally **150** epochs training with **0.01** base learning rate and **multi-steps** learning strategy.  
Learning rate reduced 10% during **60%** and **85%** training.
### CNN Structure
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/images/figure%206.png" width = "113" height = "375" align=center />
</div>  

### Training loss
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/Train/data/loss.png" width = "400" height = "300" align=center />
</div>   
 
### Testing accuracy
Highest test accuracy on FER2013: **78%**.

<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/Train/data/accuracy.png" width = "400" height = "300" align=center />
</div>  
<br>
<br>
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/images/figure%209.png" width = "400" height = "300" align=center />
</div> 
<br>
<br>
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/images/figure%204.png" width = "377" height = "157" align=center />
</div> 


## Predicting
### Preprocess
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/images/figure%203.png" width = "600" height = "230" align=center />
</div> 

### Predict
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/images/figure%205.png" width = "601" height = "115" align=center />
</div> 

## References
[A Real-time Facial Expression Recognizer using Deep Neural Network](http://brain.kaist.ac.kr/document/JJW/ACM_IMCOM_2016_JJW.pdf)  


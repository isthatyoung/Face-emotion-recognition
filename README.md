# Face-emotion-recognition
An implementation of face emotion recognition by Convolutional neurla network.

## Requirements
* Caffe 1.0.0
* Python 2.7
* OpenCV (cv2)
* Numpy
* skimage
* [mtcnn](https://github.com/ipazc/mtcnn)

## Description
As humans, we classify emotions all the time without knowing it. 
We can see if someone is happy or sad or frustrated and in need of help. 
However, it is a very complex problem that involves many subtleties about facial expression. 
Even just the tiniest change in someone’s face can be a signal of a different emotion. 
Training models that understand human emotions will be critical to build truly intelligent machine that can interact with us humans do. 

In this project, we build a classifier to detect basic human emotion from facial expression by two steps: face detection and CNN based recognition. 

## Datasets
### Training
**FER2013 database** from Kaggle
*Challenges in Representation Learning: Facial Expression Recognition Challenge*  
 
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/images/Figure%201.png" width = "359" height = "169" align=center />
</div>  
This dataset has 7 facial expression categories (angry, disgust, fear, happy, sad, surprise and neutral)<br>  
Totally 28709 training images, 3589 validation images and 3589 test images with grayscale image size 48px * 48px.  

### Testing
* Testing images of FER2013  

## Main tasks
* Preprocessing (recent accomplished)
1. Extract and create images for FER2013 from csv
2. Face detection
3. Face alignment
4. Convert to LMDB
5. Compute mean

* Training & Testing (recent accomplished)
1. Build model
2. Train model
3. Adjust hyper-parameters

* Predicting (recent accomplished)

## Preprocessing
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/images/Figure%202.png" width = "311" height = "125" align=center />
</div>  

## Training & Testing
### CNN Structure
### Training loss
<div align="center">
<img src="https://github.com/isthatyoung/Face-emotion-recognition/blob/master/Train/data/loss.png" width = "400" height = "300" align=center />
</div>   
 
### Testing accuracy
Highest test accuracy on FER2013: **69%**.
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
[The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression](http://www.pitt.edu/~jeffcohn/CVPR2010_CK+2.pdf)

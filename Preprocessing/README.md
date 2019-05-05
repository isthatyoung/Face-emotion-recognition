## Instruction
**Tips** I put the repository under /home/ubuntu/, so all the dictionary path in my code is /home/ubuntu/Face-emotion-recognition-master/XXX. 
### Preprocessing
1. Download fer2013.csv from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Put it into /Preprocessing/data/fer2013
2. Run  /Preprocessing/code/FER2013_extraction_from_csv.py
3. Prepare for MTCNN environment
```
pip3 install mtcnn
pip3 install tensorflow==1.4.1 opencv-contrib-python==3.2.0.
```
4. Use Python3 run /Preprocessing/code/Face-detection+alignment.py to do face detection and face alignment.
```
python3 /Preprocessing/code/Face-detection+alignment.py
```
5. Create training and testing list. Run /Preprocessing/code/FER2013_get_file_list_LMDB.py
6. 
```
chmod 777 /Preprocessing/code/convert_lmdb.sh
chmod 777 /Preprocessing/code/compute_mean.sh
/Preprocessing/code/convert_lmdb.sh
/Preprocessing/code/compute_mean.sh
```
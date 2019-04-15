#from mtcnn.mtcnn import MTCNN
import caffe
import cv2
import matplotlib
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt
#from align_faces import warp_and_crop_face, get_reference_facial_points
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def main():
    #plt.figure()
    dir = '/home/ubuntu/Face-emotion-recognition-master/'
    net = caffe_load_model(dir)
    #face_key_points = MTCNN_face_detection(dir)
    #face_align_similarity_transformation(face_key_points,dir)
    image = caffe.io.load_image(dir + 'Test/data/face-align-test.png')
    #plt.imshow(image)

    binMean = dir + 'Preprocessing/data/train_mean.binaryproto'
    npyMean = dir + 'Preprocessing/data/mean.npy'
    convert_mean(binMean, npyMean)
    im = caffe.io.load_image(dir+'Test/data/face-align-test.png',color=False)
    predict(im,net,dir)

    plt.show()


def caffe_load_model(dir):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    model = dir + 'Test/code/FaceEmotionNet_deploy.prototxt'
    caffemodel_file = dir + 'Train/data/FaceEmotionNet.caffemodel'
    net = caffe.Net(model, caffemodel_file, caffe.TEST)
    return net

def MTCNN_face_detection(dir):
    img = cv2.imread(dir + 'Test/data/test.png')
    detector = MTCNN()
    result = detector.detect_faces(img)
    print(result)
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(img,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 0, 255),
                  1)

    cv2.circle(img, (keypoints['left_eye']), 1, (0, 0, 255), 2)
    cv2.circle(img, (keypoints['right_eye']), 1, (0, 0, 255), 2)
    cv2.circle(img, (keypoints['nose']), 1, (0, 0, 255), 2)
    cv2.circle(img, (keypoints['mouth_left']), 1, (0, 0, 255), 2)
    cv2.circle(img, (keypoints['mouth_right']), 1, (0, 0, 255), 2)
    cv2.imwrite(dir+'Test/data/face-detect-test.png', img)
    return keypoints

def face_align_similarity_transformation(facial_5_points, dir):
    ##Load as RGB
    img = cv2.imread(dir + 'Test/data/test.png')
    facial_5_points = np.array([facial_5_points['left_eye'], facial_5_points['right_eye'], facial_5_points['nose'],
                                facial_5_points['mouth_left'], facial_5_points['mouth_right']], dtype=np.float32)

    default_square = True
    inner_padding_factor = 0.001
    outer_padding = (0, 0)
    output_size = (48, 48)

    reference_5pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)

    dst_img = warp_and_crop_face(img, facial_5_points, reference_pts=reference_5pts, crop_size=(48, 48))

    ##RGB to gray
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(dir+'Test/data/face-align-test.png', dst_img)



def predict(im, net, dir):

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_mean('data', np.load(dir + 'Preprocessing/data/mean.npy').mean(1).mean(1))
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    out = net.forward()
    prob = net.blobs['prob'].data[0].flatten()
    order = prob.argsort()[-1]
    if order == 0:
        emotion = 'Angry'
    elif order == 1:
        emotion = 'Disgust'
    elif order == 2:
        emotion = 'Fear'
    elif order == 3:
        emotion = 'Happy'
    elif order == 4:
        emotion = 'Sad'
    elif order == 5:
        emotion = 'Surprise'
    elif order == 6:
        emotion = 'Neutral'

    print('Probability: {}'.format(prob))
    print("Predict emotion is {}".format(emotion))



def convert_mean(binMean,npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb' ).read()
    blob.ParseFromString(bin_mean)
    arr = np.array(caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npyMean, npy_mean)










if __name__ == '__main__':
    main()

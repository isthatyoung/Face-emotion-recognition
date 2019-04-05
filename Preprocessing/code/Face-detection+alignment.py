from mtcnn.mtcnn import MTCNN
from skimage import transform as trans
import cv2
import os
import numpy as np
from align_faces import warp_and_crop_face, get_reference_facial_points
def main():
    dataset = 'fer2013'
    path = '/home/ubuntu/face-emotion-detection/data/' + dataset
    dirs = os.listdir(path)
    detector = MTCNN()
    if dataset == 'CK+48':
        for dir in dirs:
            for image in os.listdir('{}/{}'.format(path,dir)):
                img = cv2.imread('{}/{}/{}'.format(path, dir, image))
                facial_5_points = MTCNN_face_detection(img,detector,dir,image, dataset)
                img = cv2.imread('{}/{}/{}'.format(path, dir, image))
                face_align_similarity_transformation(img, facial_5_points, dir,image, dataset)
    else:
        path = '/home/ubuntu/face-emotion-detection/data/' + dataset
        for dir in dirs:
            for label_dir in os.listdir('{}/{}'.format(path,dir)):
                for image in os.listdir('{}/{}/{}'.format(path,dir,label_dir)):
                    img = cv2.imread('{}/{}/{}/{}'.format(path, dir, label_dir, image))
                    facial_5_points = MTCNN_face_detection(img, detector, dir+'/'+label_dir, image, dataset)
                    img = cv2.imread('{}/{}/{}/{}'.format(path, dir, label_dir, image))
                    face_align_similarity_transformation(img, facial_5_points, dir+'/'+label_dir, image, dataset)






def MTCNN_face_detection(img, detector, dir_path, image_path, dataset):
    result = detector.detect_faces(img)
    if len(result) != 0:
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

        folder = os.path.exists("/home/ubuntu/face-emotion-detection/data/face-detection-{}/{}".format(dataset,dir_path))
        if not folder:
            os.makedirs("/home/ubuntu/face-emotion-detection/data/face-detection-{}/{}".format(dataset,dir_path))
        cv2.imwrite("/home/ubuntu/face-emotion-detection/data/face-detection-{}/{}/{}".format(dataset,dir_path, image_path), img)

    else:
        keypoints = None
        folder = os.path.exists("/home/ubuntu/face-emotion-detection/data/face-detection-{}/{}".format(dataset,dir_path))
        if not folder:
            os.makedirs("/home/ubuntu/face-emotion-detection/data/face-detection-{}/{}".format(dataset,dir_path))
        cv2.imwrite("/home/ubuntu/face-emotion-detection/data/face-detection-{}/{}/{}".format(dataset,dir_path, image_path), img)

    return keypoints

def face_align_similarity_transformation(img, facial_5_points, dir_path, image_path, dataset):
    if facial_5_points != None:
        print(facial_5_points)
        # #facial_5_points = np.array(facial_5_points, dtype=np.float32)
        facial_5_points = np.array([facial_5_points['left_eye'], facial_5_points['right_eye'], facial_5_points['nose'], facial_5_points['mouth_left'], facial_5_points['mouth_right']],dtype=np.float32)

        #
        # coord_5_points = np.array([
        #     [30.2946, 51.6963],
        #     [65.5318, 51.5014],
        #     [48.0252, 71.7366],
        #     [33.5493, 92.3655],
        #     [62.7299, 92.2041]], dtype=np.float32)
        #
        # transform = trans.SimilarityTransform()
        # transform.estimate(facial_5_points, coord_5_points)
        # M = transform.params[0:2,:]
        # width = img.shape[1]
        # height = img.shape[0]
        # warped = cv2.warpAffine(img, M, (48,48), borderValue=(255,255,255))

        default_square = True
        inner_padding_factor = 0.001
        outer_padding = (0, 0)
        output_size = (48,48)

        reference_5pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)

        dst_img = warp_and_crop_face(img, facial_5_points, reference_pts=reference_5pts, crop_size=(48,48))
        # transform = trans.SimilarityTransform()
        # transform.estimate(facial_5_points, reference_5pts)
        # M = transform.params[0:2,:]
        #
        # warped = cv2.warpAffine(img, M, output_size, borderValue=0)

        ##RGB to gray
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
        folder = os.path.exists("/home/ubuntu/face-emotion-detection/data/face-alignment-{}/{}".format(dataset,dir_path))
        if not folder:
            os.makedirs("/home/ubuntu/face-emotion-detection/data/face-alignment-{}/{}".format(dataset,dir_path))
        cv2.imwrite("/home/ubuntu/face-emotion-detection/data/face-alignment-{}/{}/{}".format(dataset,dir_path, image_path), dst_img)

    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        folder = os.path.exists("/home/ubuntu/face-emotion-detection/data/face-alignment-{}/{}".format(dataset,dir_path))
        if not folder:
            os.makedirs("/home/ubuntu/face-emotion-detection/data/face-alignment-{}/{}".format(dataset,dir_path))
        cv2.imwrite("/home/ubuntu/face-emotion-detection/data/face-alignment-{}/{}/{}".format(dataset,dir_path, image_path), img)







main()
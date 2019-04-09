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
    face_count = 0
    count = 0
    if dataset == 'CK+48':
        for dir in dirs:
            for image in os.listdir('{}/{}'.format(path,dir)):
                img = cv2.imread('{}/{}/{}'.format(path, dir, image))
                facial_5_points, face_count = MTCNN_face_detection(img,detector,dir,image, dataset, face_count)
                img = cv2.imread('{}/{}/{}'.format(path, dir, image))
                count = face_align_similarity_transformation(img, facial_5_points, dir,image, dataset, count)
    else:
        path = '/home/ubuntu/face-emotion-detection/data/' + dataset
        for dir in dirs:
            for label_dir in os.listdir('{}/{}'.format(path,dir)):
                for image in os.listdir('{}/{}/{}'.format(path,dir,label_dir)):
                    img = cv2.imread('{}/{}/{}/{}'.format(path, dir, label_dir, image))
                    facial_5_points, face_count = MTCNN_face_detection(img, detector, dir+'/'+label_dir, image, dataset, face_count)
                    img = cv2.imread('{}/{}/{}/{}'.format(path, dir, label_dir, image))
                    count = face_align_similarity_transformation(img, facial_5_points, dir+'/'+label_dir, image, dataset, count)

    print("Processed totally {} images.".format(count))
    print("Totally {} images detected with face".format(face_count))





def MTCNN_face_detection(img, detector, dir_path, image_path, dataset, count):
    result = detector.detect_faces(img)
    if len(result) != 0:
        count += 1
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

    return keypoints, count

def face_align_similarity_transformation(img, facial_5_points, dir_path, image_path, dataset, count):
    count += 1
    if facial_5_points != None:
        print(facial_5_points)
        # #facial_5_points = np.array(facial_5_points, dtype=np.float32)
        facial_5_points = np.array([facial_5_points['left_eye'], facial_5_points['right_eye'], facial_5_points['nose'], facial_5_points['mouth_left'], facial_5_points['mouth_right']],dtype=np.float32)

        default_square = True
        inner_padding_factor = 0.001
        outer_padding = (0, 0)
        output_size = (48,48)

        reference_5pts = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)

        dst_img = warp_and_crop_face(img, facial_5_points, reference_pts=reference_5pts, crop_size=(48,48))

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

    return count






main()
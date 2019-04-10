##TODO Create .txt for images file list with full path
import os

def main():
    path = '/home/ubuntu/face-emotion-detection/data/'
    f_train = open(path + 'train_list.txt', 'w+')
    f_validation = open(path + 'validation_list.txt', 'w+')
    f_test = open(path + 'test_list.txt', 'w+')
    train_dir = os.listdir(path + 'fer2013/train')
    val_dir = os.listdir(path + 'fer2013/val')
    test_dir = os.listdir(path + 'fer2013/test')

    for dir in train_dir:
        for image in os.listdir('{}/{}'.format(path + 'fer2013/train', dir)):
            f_train.writelines('{}face-alignment-fer2013/train/{}/{} {}'.format(path, dir, image, dir))
            f_train.write('\n')

    for dir in val_dir:
        for image in os.listdir('{}/{}'.format(path + 'fer2013/val', dir)):
            f_validation.writelines('{}face-alignment-fer2013/val/{}/{} {}'.format(path, dir, image, dir))
            f_validation.write('\n')

    for dir in test_dir:
        for image in os.listdir('{}/{}'.format(path + 'fer2013/test', dir)):
            f_validation.writelines('{}face-alignment-fer2013/test/{}/{} {}'.format(path, dir, image, dir))
            f_validation.write('\n')

    f_train.close()
    f_validation.close()
    f_test.close()



if __name__ == '__main__':
    main()
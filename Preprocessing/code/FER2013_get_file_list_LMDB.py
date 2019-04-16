##TODO Create .txt for images file list with relative path for lmdb convert script
import os


def main():
    path = '/home/ubuntu/Face-emotion-recognition-master/Preprocessing/data/'
    f_train = open(path + 'train_list.txt', 'w')
    #f_validation = open(path + 'validation_list.txt', 'w')
    f_test = open(path + 'test_list.txt', 'w')
    train_dir = os.listdir(path + 'fer2013/train')
    val_dir = os.listdir(path + 'fer2013/val')
    test_dir = os.listdir(path + 'fer2013/test')

    for dir in train_dir:
        for image in os.listdir('{}/{}'.format(path + 'fer2013/train', dir)):
            f_train.writelines('/train/{}/{} {}'.format(dir,image, dir))
            f_train.write('\n')

    for dir in val_dir:
        for image in os.listdir('{}/{}'.format(path + 'fer2013/val', dir)):
            f_train.writelines('/val/{}/{} {}'.format(dir, image, dir))
            f_train.write('\n')


    for dir in test_dir:
        for image in os.listdir('{}/{}'.format(path + 'fer2013/test', dir)):
            f_test.writelines('/test/{}/{} {}'.format(dir,image, dir))
            f_test.write('\n')

    f_train.close()
    #f_validation.close()
    f_test.close()



if __name__ == '__main__':
    main()
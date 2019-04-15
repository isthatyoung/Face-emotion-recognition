import os
def main():
    
    path = '/home/ubuntu/Face-emotion-recognition-master/Preprocessing/data/'
    f_train = open(path+'CK48_train_list.txt', 'w+')
    f_test = open(path+'CK48_test_list.txt', 'w+')
    dirs = os.listdir(path+'CK+48')
    for dir in dirs:
        count = 0
        for image in os.listdir('{}/{}'.format(path+'CK+48', dir)):
            num_image = len(os.listdir('{}/{}'.format(path+'CK+48', dir)))
            if count >= 0.8*num_image:
                f_test.writelines('/{}/{} {}'.format(dir, image, dir))
                f_test.write('\n')
            else:
                f_train.writelines('/{}/{} {}'.format(dir, image, dir))
                f_train.write('\n')
                count += 1
    f_train.close()
    f_test.close()

if __name__ == '__main__':
    main()
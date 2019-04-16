import os
import re
def main():
    
    path = '/home/ubuntu/Face-emotion-recognition-master/Preprocessing/data/'
    #f_train = open(path+'CK_train_list.txt', 'w+')
    f_train = open(path+'CK_train_list.txt', 'w+')
    f_test = open(path+'CK_test_list.txt', 'w+')
    dirs = os.listdir(path+'Emotion')
    num_image = 327
    count = 0
    for first_dir in dirs:
        if not first_dir.startswith('.'):
            for second_dir in os.listdir('{}/{}'.format(path+'Emotion', first_dir)):
                if not second_dir.startswith('.'):
                    # num_image = len(os.listdir('{}/{}'.format(path+'Emotion', dir)))
                    for file in os.listdir('{}/{}/{}'.format(path+'Emotion', first_dir, second_dir)):
                        if not file.startswith('.'):
                            pattern = r'(.+?)\.'
                            file_name = re.findall(pattern,file)[0].strip('_emotion')
                            f = open('{}/{}/{}/{}'.format(path+'Emotion', first_dir, second_dir, file),'r')
                            for line in f.readlines():
                                line = line.strip('\n')
                                line = line.strip()
                                print(line[0])
                                if line[0] != '2':
                                    if line[0] == '7':
                                        line = 5
                                    elif line[0] == '6':
                                        line = 4
                                    elif line[0] == '5':
                                        line = 3
                                    elif line[0] == '4':
                                        line = 2
                                    elif line[0] == '3':
                                        line = 1
                                    elif line[0] == '1':
                                        line = 0

                                    f_test.writelines('/{}/{}/{}.png {}'.format(first_dir, second_dir, file_name, line))
                                    f_test.writelines('\n')
                                    count += 1

    f_test.close()
if __name__ == '__main__':
    main()
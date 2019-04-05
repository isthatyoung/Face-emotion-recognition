# -*- coding: utf-8 -*-
import csv
import os
from PIL import Image
import numpy as np

def main():
    path = '/home/ubuntu/face-emotion-detection/data/'
    database_path = os.path.join(os.path.expanduser("~"), path+'fer2013')
    datasets_path = os.path.join(os.path.expanduser("~"), path+'fer2013')

    csv_file = os.path.join(database_path, "fer2013.csv")
    train_csv = os.path.join(datasets_path, 'train.csv')
    val_csv = os.path.join(datasets_path, 'val.csv')
    test_csv = os.path.join(datasets_path, 'test.csv')

    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        rows = [row for row in csvr]

        trn = [row[:-1] for row in rows if row[-1] == 'Training']
        csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
        print(len(trn))

        val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
        csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
        print(len(val))

        tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
        print(len(tst))

    train_set = os.path.join(datasets_path, 'train')
    val_set = os.path.join(datasets_path, 'val')
    test_set = os.path.join(datasets_path, 'test')

    for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        num = 1
        with open(csv_file) as f:
            csvr = csv.reader(f)
            header = next(csvr)
            for i, (label, pixel) in enumerate(csvr):
                pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
                subfolder = os.path.join(save_path, label)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                im = Image.fromarray(pixel).convert('L')
                image_name = os.path.join(subfolder, '{:05d}.png'.format(i))
                print(image_name)
                im.save(image_name)



if __name__ == '__main__':
    main()
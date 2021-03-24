import os
import numpy as np
from shutil import copyfile, rmtree

RAW_DATA_FOLDER = './data/part_classifier/raw/'

TRAINING_FOLDER = './data/part_classifier/training/'
VAL_RATIO = 0.3

def init_data_training_folder(training_folder, classes):
    rmtree(training_folder)
    os.mkdir(training_folder)
    os.mkdir(training_folder + 'train')
    os.mkdir(training_folder + 'val')

    for c in classes:
        os.mkdir(training_folder + 'train/' + c)
        os.mkdir(training_folder + 'val/' + c)

def drop_file(file_path):
    file_name = file_path.split('/')[-1]
    item_class = file_path.split('/')[-2]

    destination = TRAINING_FOLDER + ('val/' if np.random.random() < VAL_RATIO else 'train/') + item_class + '/' + file_name
    copyfile(file_path, destination)

def recursion_get_all_files(root_folder):
    files = []
    for f in os.listdir(root_folder):
        if os.path.isdir(root_folder + f):
            files.extend(recursion_get_all_files(root_folder + f + '/'))
        else:
            files.append(root_folder + f)
    return files

if __name__ == '__main__':
    init_data_training_folder(TRAINING_FOLDER, os.listdir(RAW_DATA_FOLDER))

    for idx, f in enumerate(recursion_get_all_files(RAW_DATA_FOLDER)):
        drop_file(f)
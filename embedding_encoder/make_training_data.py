import os
import numpy as np
import pandas as pd
from shutil import copyfile, rmtree


MIN_IMAGE_COUNT = 100
DATABASE_CSV = './data/images_db.csv'
TRAINING_FOLDER = './data/embedding_encoder/training/'
VAL_RATIO = 0.3
FOCUS_PART = 'Flower'


def get_selected_items(images_df: pd.DataFrame):
    images_df = images_df.copy()
    images_df = images_df[images_df['content_type'] == FOCUS_PART]

    grouped = images_df.groupby('spiece_id')['image_name'].nunique()
    keys = grouped.keys().to_numpy()
    vals = grouped.values
    keys = keys[vals > MIN_IMAGE_COUNT]

    images_df = images_df[images_df['spiece_id'].isin(keys)]

    return images_df


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


if __name__ == '__main__':
    images_df = pd.read_csv(DATABASE_CSV)
    images_df = get_selected_items(images_df)
    images_df['spiece_id'] = images_df['spiece_id'].apply(lambda x: str(x))
    classes = images_df['spiece_id'].unique().tolist()
    init_data_training_folder(TRAINING_FOLDER, classes)

    for idx, f in enumerate(images_df['image_path'].tolist()):
        print('Processing file {} of {}'.format(idx+1, len(images_df)), end='\r')
        drop_file(f[1:])

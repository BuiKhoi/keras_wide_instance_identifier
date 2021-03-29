import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence


class TripletGenerator(Sequence):
    def __init__(self, image_directory, batch_size, image_dims, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_dir = image_directory
        self.dims = image_dims

        self.classes = len(os.listdir(self.img_dir))
        self.X_images_data = []
        self.y_label = []
        self.init_data()

        print('Found {} images belongs to {} classes'.format(len(self.X_images_data), self.classes))

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(self.X_images_data)//3

    def __getitem__(self, index):
        [A, P, N], label = self.generate_triplets()
        A = self.load_images(A)
        P = self.load_images(P)
        N = self.load_images(N)

        return [A, P, N], label

    def load_images(self, image_list):
        images = []
        for il in image_list:
            image = cv2.imread(il)[:, :, (2, 1, 0)]
            image = cv2.resize(image, (self.dims[1], self.dims[0]))
            images.append(image)
        return np.array(images, np.float32)

    def init_data(self):
        X_images_data = []
        y_label = []
        for idx, class_name in enumerate(os.listdir(self.img_dir)):
            class_path = self.img_dir + class_name + '/'

            for image in os.listdir(class_path):
                image_path = class_path + image
                X_images_data.append(image_path)
                y_label.append(idx)

        self.X_images_data = np.array(X_images_data)
        self.y_label = np.array(y_label)

    def search_image_with_label(self, label):
        """Choose an image from our training or test data with the
        given label."""
        y = self.y_label
        X = self.X_images_data
        poss = np.argwhere(y == label)
        idx = np.random.choice([p[0] for p in poss])
        return X[idx]

    def get_triplet(self):
        """Choose a triplet (anchor, positive, negative) of images
        such that anchor and positive have the same label and
        anchor and negative have different labels."""
        n = a = -1
        while n == a:
            n, a = np.random.choice(self.classes, 2)
        a, p = self.search_image_with_label(a), self.search_image_with_label(a)
        n = self.search_image_with_label(n)
        return a, p, n

    def generate_triplets(self):
        """Generate an un-ending stream (ie a generator) of triplets for
        training or test."""
        list_a = []
        list_p = []
        list_n = []

        for i in range(self.batch_size):
            a, p, n = self.get_triplet()
            list_a.append(a)
            list_p.append(p)
            list_n.append(n)
        # a "dummy" label which will come in to our identity loss
        # function below as y_true. We'll ignore it.
        label = np.ones(self.batch_size)
        return [list_a, list_p, list_n], label

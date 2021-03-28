import numpy as np
from tensorflow.keras.models import Model
import cv2

from embedding_encoder import embedding_model_config
from embedding_encoder.models import construct_model


class InstanceIdentifier:
    def __init__(self, model_link=None):
        test_model = construct_model(69, embedding_model_config.NUM_FEATURES)
        if not model_link:
            model_link = './embedding_encoder/checkpoints/model_cos_69.hdf5'
        test_model.load_weights(model_link)

        self.test_model = Model(inputs=test_model.input[0], outputs=test_model.layers[-3].output)

    def extract_embeddings(self, images_links):
        images = [cv2.imread(image_link)[:, :, (2, 1, 0)] for image_link in images_links]
        images = np.array([cv2.resize(image, (224, 224)) for image in images])
        embedded_features = self.test_model.predict(images)
        embedded_features /= np.linalg.norm(embedded_features, axis=1, keepdims=True)

        return embedded_features

    def extract_all_embeddings(self, x_images):
        xx = []
        for idx in range(0, len(x_images), 10):
            print('Processing at {} on {}'.format(idx, len(x_images)), end='\r')
            xx.extend(self.extract_embeddings(x_images[idx: idx + 10]))
        return xx


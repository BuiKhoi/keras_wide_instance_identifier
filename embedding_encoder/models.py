import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from embedding_encoder.docs.keras_arcface.metrics import *

weight_decay = 1e-4


def construct_model(classes, num_features):
    y = Input(shape=(classes,))

    base_model = MobileNetV2(include_top=False, pooling='max', input_shape=(224, 224, 3), weights=None)
    base_outp = base_model.output

    dns1 = Dense(256, activation='relu')(base_outp)
    dns1 = Dropout(0.2)(dns1)

    dns2 = Dense(128, activation='relu')(dns1)
    dns2 = Dropout(0.2)(dns2)

    dns3 = Dense(num_features, kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(dns2)

    outp = BatchNormalization()(dns3)
    outp = ArcFace(classes, regularizer=regularizers.l2(weight_decay))([outp, y])

    return Model([base_model.input, y], outp)

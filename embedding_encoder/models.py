import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet50 import ResNet50
from embedding_encoder.docs.keras_arcface.metrics import *

weight_decay = 1e-4


def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x


def vgg8_cosface(classes, num_features):
    input = Input(shape=(224, 224, 3))
    y = Input(shape=(classes,))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = CosFace(classes, regularizer=regularizers.l2(weight_decay))([x, y])

    return Model([input, y], output)


def construct_model(classes, num_features):
    y = Input(shape=(classes,))

    base_model = ResNet50(include_top=False, pooling='max', input_shape=(224, 224, 3), weights=None)
    base_outp = base_model.output

    dns1 = Dense(256, activation='relu')(base_outp)
    dns1 = Dropout(0.2)(dns1)

    dns2 = Dense(128, activation='relu')(dns1)
    dns2 = Dropout(0.2)(dns2)

    dns3 = Dense(num_features, kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(dns2)

    outp = BatchNormalization()(dns3)
    outp = CosFace(classes, regularizer=regularizers.l2(weight_decay))([outp, y])

    return Model([base_model.input, y], outp)


def triplet_model(num_features):
    model = tf.keras.Sequential([
        ResNet50(include_top=False, pooling='max', input_shape=(224, 224, 3), weights=None),
        tf.keras.layers.Dense(256, activation='relu'),
        Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        Dropout(0.2),
        tf.keras.layers.Dense(num_features, activation=None),  # No activation on final dense layer
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
    ])
    return model

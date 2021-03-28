from embedding_encoder.embedding_model_config import *
from tensorflow.keras.optimizers import SGD, Adam
from embedding_encoder.models import construct_model, vgg8_cosface, triplet_model
from embedding_encoder.ee_image_generator import EmbeddingDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN
from embedding_encoder.docs.keras_arcface.scheduler import CosineAnnealingScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from embedding_encoder.losses import triplet_loss_adapted_from_tf as triplet_loss
import tensorflow as tf


def prepare_datagen_custom_loss():
    train_gen = EmbeddingDataGenerator(TRAINING_FOLDER + 'train', BATCH_SIZE)

    val_gen = EmbeddingDataGenerator(TRAINING_FOLDER + 'val', BATCH_SIZE)
    return train_gen, val_gen


def prepare_datagen_triplet():
    train_gen = ImageDataGenerator(
        rotation_range=10,
        shear_range=0.3,
        zoom_range=0.3,
        rescale=1.0/255.0
    )
    train_gen = train_gen.flow_from_directory(
        TRAINING_FOLDER + 'train',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='sparse'
    )

    val_gen = ImageDataGenerator(
        rotation_range=10,
        shear_range=0.3,
        zoom_range=0.3,
        rescale=1.0/255.0
    )
    val_gen = val_gen.flow_from_directory(
        TRAINING_FOLDER + 'val',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='sparse'
    )
    return train_gen, val_gen


def load_pretrained(new_model: tf.keras.models.Model):
    if LOSS_FUNC == 'triplet':
        pass
    else:
        old_model = construct_model(36, NUM_FEATURES)
        old_model.load_weights(LOAD_WEIGHT)
        for layer in old_model.layers:
            new_model.set_weights(layer.get_weights())
        return new_model


def train_model():
    optimizer = Adam(lr=LEARNING_RATE)
    if OPTIMIZER == 'sgd':
        optimizer = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)

    if LOSS_FUNC == 'triplet':
        train_gen, val_gen = prepare_datagen_custom_loss()
        model = triplet_model(NUM_FEATURES)
        model.compile(
            optimizer=optimizer,
            loss=triplet_loss)
    else:
        train_gen, val_gen = prepare_datagen_custom_loss()
        model = construct_model(train_gen.num_classes, NUM_FEATURES)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
    if LOAD_MODEL:
        load_pretrained(model)
    model.summary()

    callbacks = [
        ModelCheckpoint(CHECKPOINT_FOLDER + 'model.hdf5',
                        verbose=1, save_best_only=True),
        CSVLogger(CHECKPOINT_FOLDER + 'log.csv'),
        TerminateOnNaN()]

    if SCHEDULER == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(
            T_max=EPOCHS,
            eta_max=LEARNING_RATE,
            eta_min=MIN_LEARNING_RATE,
            verbose=1
        ))

    model.fit(train_gen, validation_data=val_gen,
              epochs=EPOCHS,
              callbacks=callbacks,
              verbose=1)


if __name__ == '__main__':
    train_model()

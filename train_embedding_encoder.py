from embedding_encoder.embedding_model_config import *
from tensorflow.keras.optimizers import SGD, Adam
from embedding_encoder.models import construct_model, triplet_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN
from embedding_encoder.docs.keras_arcface.scheduler import CosineAnnealingScheduler
from embedding_encoder.losses import identity_loss
import tensorflow as tf


def prepare_datagen_custom_loss():
    from embedding_encoder.ee_image_generator import EmbeddingDataGenerator
    train_gen = EmbeddingDataGenerator(TRAINING_FOLDER + 'train', BATCH_SIZE)

    val_gen = EmbeddingDataGenerator(TRAINING_FOLDER + 'val', BATCH_SIZE)
    return train_gen, val_gen


def prepare_datagen_triplet():
    from embedding_encoder.triplet_generator import TripletGenerator
    train_gen = TripletGenerator(TRAINING_FOLDER + 'train/', BATCH_SIZE, IMAGE_SHAPE)
    val_gen = TripletGenerator(TRAINING_FOLDER + 'val/', BATCH_SIZE, IMAGE_SHAPE)

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
        train_gen, val_gen = prepare_datagen_triplet()
        model = triplet_model(IMAGE_SHAPE, NUM_FEATURES)
        model.compile(
            optimizer=optimizer,
            loss=identity_loss)
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

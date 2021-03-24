from embedding_encoder.embedding_model_config import *
from tensorflow.keras.optimizers import SGD, Adam
from embedding_encoder.models import construct_model
from embedding_encoder.ee_image_generator import EmbeddingDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN
from embedding_encoder.docs.keras_arcface.scheduler import CosineAnnealingScheduler


def prepare_datagen():
    train_gen = EmbeddingDataGenerator(TRAINING_FOLDER + 'train', BATCH_SIZE)

    val_gen = EmbeddingDataGenerator(TRAINING_FOLDER + 'val', BATCH_SIZE)
    return train_gen, val_gen


def main():
    train_gen, val_gen = prepare_datagen()

    optimizer = Adam(lr=LEARNING_RATE)
    if OPTIMIZER == 'sgd':
        optimizer = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)

    model = construct_model(train_gen.num_classes, NUM_FEATURES)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
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

    model.load_weights(CHECKPOINT_FOLDER + 'model.hdf5')


if __name__ == '__main__':
    main()

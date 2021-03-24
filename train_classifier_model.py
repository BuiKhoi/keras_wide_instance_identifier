from parts_classifier.models import construct_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import pandas as pd

TRAINING_FOLDER = './data/part_classifier/training/'


def prepare_datagen():
    train_gen = ImageDataGenerator(rotation_range=10, shear_range=0.3, zoom_range=0.3)
    train_gen = train_gen.flow_from_directory(TRAINING_FOLDER + 'train', target_size=(224, 224), batch_size=64)

    val_gen = ImageDataGenerator(rotation_range=10, shear_range=0.3, zoom_range=0.3)
    val_gen = val_gen.flow_from_directory(TRAINING_FOLDER + 'val', target_size=(224, 224), batch_size=64)
    return train_gen, val_gen


def main():
    training_model = construct_model(5)
    train_gen, val_gen = prepare_datagen()

    training_model.compile(Adam(1e-4), 'categorical_crossentropy', metrics=['accuracy'])
    callbacks = [
        ModelCheckpoint('./parts_classifier/checkpoints/part_classifier_best.h5', verbose=1, save_best_only=True),
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=20, verbose=1)
    ]
    history = training_model.fit(
        train_gen,
        epochs=200,
        callbacks=callbacks,
        validation_data=val_gen
    )

    keys = history.history.keys()
    history_df = pd.DataFrame(columns=keys)
    for k in keys:
        history_df[k] = history.history[k]
    history_df.to_csv('./parts_classifier/checkpoints/history.csv', index=False)


if __name__ == '__main__':
    main()

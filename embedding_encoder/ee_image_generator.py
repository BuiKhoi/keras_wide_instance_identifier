from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class EmbeddingDataGenerator(Sequence):
    def __init__(self, image_directory, batch_size=32, shuffle=True):
        self.image_directory = image_directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = ImageDataGenerator(rotation_range=10, shear_range=0.3, zoom_range=0.3)
        self.generator = self.generator.flow_from_directory(
            image_directory,
            target_size=(224, 224),
            batch_size=batch_size
        )

    def on_epoch_end(self):
        self.generator.on_epoch_end()

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        x, y = self.generator[index]
        return [x, y], y

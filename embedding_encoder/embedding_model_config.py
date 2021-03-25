LOSS_FUNC = 'cosface'
OPTIMIZER = 'adam'
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-8
BATCH_SIZE = 64
NUM_FEATURES = 128
SCHEDULER = 'CosineAnnealing'
MOMENTUM = 0.5
TRAINING_FOLDER = './data/embedding_encoder/training/'
EPOCHS = 100
CHECKPOINT_FOLDER = './embedding_encoder/checkpoints/'
LOAD_MODEL = False
LOAD_WEIGHT = './embedding_encoder/checkpoints/model_cos_36.hdf5'

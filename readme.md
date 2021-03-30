# Wide instances classifier project using Keras

To begin running the project, run:
```
pip install -r requirements.txt
```

The project contains 3 parts:

## Part classifier
This is for classifying tree parts, we are focus into 5 different plant parts:
```
- Entire
- Flower
- Fruit
- Leaf
- Stem
```
Each of the parts wil be classified to improve our model's performance

To run part classifier, run:
```
python train_classifier_model.py
```
Checkpoints will be saved at [parts_classifier/checkpoints/](parts_classifier/checkpoints/)

## Embedding encoder
This is for training the embedding extractor, you can config training variables with [config file](embedding_encoder/embedding_model_config.py)

Start training embedding extractor with:
```
python train_embedding_encoder.py
```

## Instance identifier
This part is for identify which instances belongs to which class
Test it with confusion matrix and classification report with:
```
python get_image_instances.py
```
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model


def construct_model(classes):
    base_model = MobileNetV2(include_top=False, pooling='max', input_shape=(224, 224, 3), weights='imagenet')
    base_outp = base_model.output

    # flnt = Flatten()(base_outp)
    
    dns1 = Dense(256, activation='relu')(base_outp)
    dns1 = Dropout(0.2)(dns1)

    dns2 = Dense(128, activation='relu')(dns1)
    dns2 = Dropout(0.2)(dns2)

    dns2 = Dense(classes, activation='softmax')(dns2)

    return Model(base_model.input, dns2)


if __name__ == '__main__':
    model = construct_model(7)
    model.summary()
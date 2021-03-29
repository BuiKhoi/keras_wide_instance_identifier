from tensorflow.keras import backend as K


def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def triplet_loss(x, alpha=1.0):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative), axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss, 0.0)
    return loss

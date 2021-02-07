import tensorflow.keras as K


def euclidean_distance(x, y):
    sum_sqrt = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_sqrt, K.epsilon()))

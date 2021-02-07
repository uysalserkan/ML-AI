import tensorflow as tf
from tensorflow.keras.losses import Loss


class MyHuberLoss(Loss):
    '''
        model.compile(optimizer='sgd', loss=MyHuberLoss(threshold=1.02))
    '''
    # class attribute
    threshold = 1

    # initialize instance attributes
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    # compute loss
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * \
            (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)

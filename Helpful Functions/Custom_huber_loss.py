import tensorflow as tf


def my_huber_loss_with_threshold(threshold):
    # function that accepts the ground truth and predictions
    def my_huber_loss(y_true, y_pred):
        '''
            model.compile(optimizer='sgd', loss=my_huber_loss_with_threshold(threshold=1.2))
        '''
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))

        return tf.where(is_small_error, small_error_loss, big_error_loss)

    # return the inner function tuned by the hyperparameter
    return my_huber_loss


def my_huber_loss(y_true, y_pred):
    '''
        model.compile(optimizer='sgd', loss=my_huber_loss)
    '''
    threshold = 1
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)

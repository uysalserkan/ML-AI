import tensorflow.keras as K


def contrastive_loss_with_margin(margin):
    '''
    With hyperparameter.
    '''
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

# model.compile(optimizer='sgd', loss=contrastive_loss_with_margin(margin=1.2)) <- Hyperparams
# model.compile(optimizer='sgd', loss=contrastive_loss) <- standart

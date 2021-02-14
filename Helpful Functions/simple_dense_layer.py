class SimleDense(Layer):
    """
    Required:
        from tensorflow.keras.layers import Layer
    """

    def __init__(self, units=32):
        """
        Initializes the instance attributes
        """
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        """
        Create the state of the layer (weights)
        """
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
                name="kernel",
                initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
                trainable=True
                )
        b_init = tf.zeros_initializer(),
        self.b = tf.Variable(
                name="bias",
                initial_value=b_init(shape(self.units,), dtype='float32'),
                trainable=True
                )

    def call(self, inputs):
        """
        Defines the computation from inputs to outputs
        """
        return matmul(inputs, self.w) + self.b

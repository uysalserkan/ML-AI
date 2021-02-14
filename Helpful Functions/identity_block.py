class IdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name='')
        self.conv = tf.keras.layers.Conv2D(
                filters, kernel_size, padding='same'
                )
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = sefl.add([x, input_tensor])
        x = self.act(x)
        return x



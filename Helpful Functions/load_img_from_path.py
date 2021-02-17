def load_img(path):
    max_dim = 512
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)[:-1]
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape* scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :],image = tf.image.convert_image_dtype(image, tf.uint8)

    return image


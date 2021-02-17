def tensor_to_image(image):
    tensor_shape = tf.shape(tensor)
    number_elem_shape = tf.shape(tensor_shape)

    if number_elem_shape > 3:
        assert tensor_shape[0]==1
        tensor = tensor[0]

    return tf.keras.preprocessing.image.array_to_img(tensor)

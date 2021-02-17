def format_images(data):
    img = data["image"]
    img = tf.reshape(img, [-1])
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    return img, data["label"]

train_data = train_data.map(format_images)
test_data = test_data.map(format_images)

train = train_data.shuffle(buffer_size=64).batch(batch_size)

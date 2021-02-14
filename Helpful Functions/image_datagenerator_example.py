train_dataget = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        )

train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=100,
        class_mode='binary',
        target_size=(244,244),
        )



def plot_hist(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    size = range(len(acc))

    plt.plot(size, acc, 'r', "Training Accuracy")
    plt.plot(size, val_acc, 'b', "Validation Accuracy")
    plt.figure()

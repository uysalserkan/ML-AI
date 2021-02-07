import matplotlib.pyplot as plt


def plot_metrics(metric_name, title, history, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name],
             color='green', label='val_' + metric_name)
    plt.show()

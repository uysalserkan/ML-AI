def gpu_usage_limit(limit=1024):
    """
        Params:
          limit: It needs to be int format and, it defines the usage megabytes.
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    )


def gpu_usage_fraction(fraction=0.6):
    """
        Params:
          fraction: 1 -> Full usage of GPU.
    """

    tf.config.gpu.set_per_process_memory_fraction(fraction)
    tf.config.gpu.set_per_process_memory_growth(True)

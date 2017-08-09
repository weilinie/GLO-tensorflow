import numpy as np
import tensorflow as tf
import os


def prepare_dirs(config, dataset):
    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(dataset, config.load_path)
    else:
        config.model_name = "{}_{}_{}_{}_zdim{}_bs{}_zitr{}". \
            format(dataset, config.loss_type, config.g_net, config.optimizer, config.z_dim,
                   config.batch_size, config.z_iters)

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)

    if not hasattr(config, 'data_path'):
        config.data_path = os.path.join(config.data_dir, dataset)

    for dir in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if not config.is_train:
        if not hasattr(config, 'test_model_dir'):
            config.test_model_dir = os.path.join(config.test_dir, config.model_name)
        if not os.path.exists(config.test_model_dir):
            os.makedirs(config.test_model_dir)


def layer_norm(inputs):
    ndims_inputs = inputs.get_shape().ndims

    mean, var = tf.nn.moments(inputs, range(1, ndims_inputs), keep_dims=True)

    # Assume the 'neurons' axis is the last of norm_axes. This is the case for fully-connected and NHWC conv layers.
    n_neurons = inputs.get_shape().as_list()[ndims_inputs - 1]

    offset = tf.Variable(np.zeros(n_neurons, dtype='float32'), name='offset')
    scale = tf.Variable(np.ones(n_neurons, dtype='float32'), name='scale')

    # Add broadcasting dims to offset and scale (e.g. NHWC conv data)
    offset = tf.reshape(offset, [1 for _ in range(ndims_inputs - 1)] + [-1])
    scale = tf.reshape(scale, [1 for _ in range(ndims_inputs - 1)] + [-1])

    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result

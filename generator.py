import tensorflow as tf
import tensorflow.contrib.layers as tcl
from utils import layer_norm


def generator(net, z, hidden_num, output_dim, out_channels, normalize_g, is_train=True, reuse=True, n_hidden_layers=1):
    if net == 'DCGAN':
        return generatorDCGAN(z, hidden_num, output_dim, out_channels, normalize_g, is_train, reuse)
    elif net == 'MLP':
        return generatorMLP(z, output_dim, out_channels, normalize_g, is_train, reuse, n_hidden_layers=n_hidden_layers)
    else:
        raise Exception('[!] Caution! unknown generator type.')


# ---------------------------------------------------
# +++++++++++++++++++++ DCGAN +++++++++++++++++++++++
# ---------------------------------------------------
def generatorDCGAN(z, hidden_num, output_dim, out_channels, normalize_g, is_train, reuse, kern_size=5):
    '''
    Default values:
    :param reuse: True
    :param is_train: True
    :param is_batchnorm: True
    :param z: 128
    :param hidden_num: 64
    :param output_dim: 64
    :param kern_size: 5
    :param out_channels: 3
    :return:
    '''
    with tf.variable_scope("G") as vs:
        if reuse:
            vs.reuse_variables()

        if normalize_g == 'BN':
            normalizer_fn = tcl.batch_norm
            normalizer_params = {'scale': True, 'is_training': is_train}
        elif normalize_g == 'LN':
            normalizer_fn = layer_norm
            normalizer_params = None
        else:
            normalizer_fn = None
            normalizer_params = None

        fc = tcl.fully_connected(
            z, hidden_num * 8 * (output_dim / 16) * (output_dim / 16),
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.relu
        )
        output = tf.reshape(fc, [-1, output_dim / 16, output_dim / 16, hidden_num * 8])  # data_format: 'NHWC'

        output = tcl.conv2d_transpose(
            output, hidden_num * 4, kern_size, stride=2,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.relu
        )

        output = tcl.conv2d_transpose(
            output, hidden_num * 2, kern_size, stride=2,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.relu
        )

        output = tcl.conv2d_transpose(
            output, hidden_num, kern_size, stride=2,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.relu
        )

        gen_out = tcl.conv2d_transpose(
            output, out_channels, kern_size, stride=2,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.tanh
        )

    g_vars = tf.contrib.framework.get_variables(vs)
    return gen_out, g_vars


# -------------------------------------------------
# +++++++++++++++++++++ MLP +++++++++++++++++++++++
# -------------------------------------------------
def generatorMLP(z, output_dim, out_channels, normalize_g, is_train, reuse, hidden_num=512, n_hidden_layers=1):
    '''
    Default values:
    :param reuse: True
    :param is_train: True
    :param is_batchnorm: True
    :param z: 128
    :param output_dim: 64
    :param out_channels: 3 or 1
    :param hidden_num: 512
    :param n_layers: 3
    :return:
    '''
    with tf.variable_scope("G") as vs:
        if reuse:
            vs.reuse_variables()

        if normalize_g == 'BN':
            normalizer_fn = tcl.batch_norm
            normalizer_params = {'scale': True, 'is_training': is_train}
        elif normalize_g == 'LN':
            normalizer_fn = layer_norm
            normalizer_params = None
        else:
            normalizer_fn = None
            normalizer_params = None

        if n_hidden_layers < 0:
            fc = tcl.fully_connected(
                z, output_dim * output_dim * out_channels,
                activation_fn=None,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params
            )
        else:
            output = tcl.fully_connected(
                z, hidden_num,
                activation_fn=tf.nn.relu,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params
            )
            for i in range(n_hidden_layers):
                output = tcl.fully_connected(
                    output, hidden_num,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params
                )
            fc = tcl.fully_connected(
                output, output_dim * output_dim * out_channels,
                activation_fn=None
            )

        gen_out = tf.reshape(fc, [-1, output_dim, output_dim, out_channels])

    g_vars = tf.contrib.framework.get_variables(vs)
    return gen_out, g_vars

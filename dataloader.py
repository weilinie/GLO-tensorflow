import os
from PIL import Image
from glob import glob
import tensorflow as tf


def load_dataset(data_path, batch_size, scale_size, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(data_path)
    if dataset_name in ['CelebA'] and split:
        data_path = os.path.join(data_path, 'splits', split)
    elif dataset_name in ['RenderBall', 'RenderBallTri']:
        data_path = data_path
    else:
        is_grayscale = True
        raise Exception('[!] Caution! Unknown dataset name.')

    paths = []
    tf_decode = tf.image.decode_jpeg
    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(data_path, ext))

        if ext == 'png':
            tf_decode = tf.image.decode_png

        if len(paths) != 0:
            break

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
        shape = [h, w, 1]
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    else:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    return tf.to_float(queue)


def load_mnist(data_path):
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    mnist_data = read_data_sets(data_path, one_hot=True)

    return mnist_data
import tensorflow as tf
import numpy as np
from dataloader import load_dataset
from generator import generator


class GLO_model(object):
    def __init__(self, config):
        self.g_net = config.g_net
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.split = config.split
        self.n_hidden_layers = config.n_hidden_layers

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.loss_type = config.loss_type

        self.z_dim = config.z_dim
        self.z_iters = config.z_iters
        self.conv_hidden_num = config.conv_hidden_num
        self.img_dim = config.img_dim
        self.g_lr = config.g_lr
        self.z_lr = config.z_lr
        self.normalize_g = config.normalize_g

        self.model_dir = config.model_dir
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.max_step = config.max_step

        self.is_train = config.is_train

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.build_model()

        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        self.sv = tf.train.Supervisor(
            logdir=self.model_dir,
            summary_op=None,
            summary_writer=self.summary_writer,
            global_step=self.global_step,
            save_model_secs=300)

    def build_model(self):
        self.x = load_dataset(
            data_path=self.data_path,
            batch_size=self.batch_size,
            scale_size=self.img_dim,
            split=self.split
        )
        img_chs = self.x.get_shape().as_list()[-1]
        x = self.x / 127.5 - 1.  # Normalization
        print("Successfully loaded {} with size: {}".format(self.dataset, self.x.get_shape()))

        # initialize z within ball(1, z_dim, 2)
        self.z = tf.Variable(tf.random_normal([self.batch_size, self.z_dim],
                                              stddev=np.sqrt(1.0/self.z_dim)), name='noise')

        fake_data, g_vars = generator(
            self.g_net, self.z, self.conv_hidden_num,
            self.img_dim, img_chs, self.normalize_g, reuse=False,
            n_hidden_layers=self.n_hidden_layers
        )

        self.fake_data = tf.clip_by_value((fake_data + 1) * 127.5, 0, 255)  # Denormalization

        x_flat = tf.reshape(self.x, [self.batch_size, -1])
        fake_data_flat = tf.reshape(self.fake_data, [self.batch_size, -1])
        if self.loss_type == 'l2':
            self.loss = tf.norm(x_flat - fake_data_flat, axis=1)
            self.loss_mean = tf.reduce_mean(self.loss)

        if self.optimizer == 'adam':
            optim_op = tf.train.AdamOptimizer
        elif self.optimizer == 'rmsprop':
            optim_op = tf.train.RMSPropOptimizer
        else:
            raise Exception("[!] Caution! Other optimizers do not apply right now!")

        self.z_optim = optim_op(self.z_lr, self.beta1, self.beta2).minimize(
            self.loss, var_list=self.z
        )
        self.g_optim = optim_op(self.g_lr, self.beta1, self.beta2).minimize(
            self.loss_mean, global_step=self.global_step, var_list=g_vars
        )

        # project z after each update to the representation space Z
        z_proj = tf.divide(self.z, tf.maximum(tf.norm(self.z, axis=1, keep_dims=True), 1))
        self.proj_op = tf.assign(self.z, z_proj)

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss/d_loss", self.loss_mean),
        ])

    def train(self):
        print('start training...\n [{}] using g_net [{}] with loss type [{}]\n'
              'batch size: {}, normalize_g: {}'.format(
               self.dataset, self.g_net, self.loss_type, self.batch_size, self.normalize_g
        ))

        with self.sv.managed_session() as sess:
            for _ in range(self.max_step):
                if self.sv.should_stop():
                    break

                step = sess.run(self.global_step)

                # Train latent variables
                for _ in range(self.z_iters):
                    sess.run([self.z_optim, self.proj_op])

                # Train generator params
                sess.run(self.g_optim)

                if step % self.log_step == 0:
                    loss_mean, summary_str = sess.run(
                        [self.loss_mean, self.summary_op]
                    )
                    self.summary_writer.add_summary(summary_str, step)
                    self.summary_writer.flush()

                    print("[{}/{}] Loss_mean: {:.6f}".format(step, self.max_step, loss_mean))

    def test(self):
        pass

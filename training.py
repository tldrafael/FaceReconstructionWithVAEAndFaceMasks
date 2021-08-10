import os
import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import utils as ut


class ConvBlock:
    def __init__(self, n_filters=64, filter_size=(3, 3), strides=(1, 1), padding='same', activation='elu', use_bn=True):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn

    def __call__(self, x):
        x = tf.keras.layers.Conv2D(self.n_filters, self.filter_size, strides=self.strides, padding=self.padding)(x)
        if self.use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation(self.activation)(x)


class ConvBlockTranspose:
    def __init__(self, n_filters=64, filter_size=(3, 3), strides=(2, 2), padding='same', activation='elu', use_bn=True):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn

    def __call__(self, x):
        x = tf.keras.layers.Conv2DTranspose(self.n_filters, self.filter_size, strides=self.strides,
                                            padding=self.padding)(x)
        if self.use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation(self.activation)(x)


class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self, add_mae=False):
        super(SSIMLoss, self).__init__(name='ssim')
        self.add_mae = add_mae
        self.tf_mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def __call__(self, y_e, y_pred, sample_weight=None):
        loss_ssim = 2 - tf.image.ssim(y_e, y_pred, 1., filter_size=9)
        loss_ssim = tf.math.reduce_mean(loss_ssim)
        if self.add_mae:
            loss_ssim = loss_ssim + 5 * self.tf_mae(y_e, y_pred)

        return tf.math.reduce_mean(loss_ssim)


class BCELoss(tf.keras.losses.Loss):
    def __init__(self):
        self.tf_bce = tf.keras.losses.BinaryCrossentropy()

    def dice_coef(self, y_e, y_pred, smooth=1):
        intersection = tf.math.abs(y_e * y_pred)
        intersection = tf.math.reduce_sum(intersection)
        total_area = tf.math.square(y_e) + tf.math.square(y_pred)
        total_area = tf.math.reduce_sum(total_area)
        return 1 - (2. * intersection + smooth) / (total_area + smooth + 1e-8)

    def __call__(self, y_e, y_pred):
        return self.tf_bce(y_e, y_pred) + self.dice_coef(y_e, y_pred)


def decay_schedule(epoch, learning_rate):
    if epoch > 1 and epoch % 9 == 0 and learning_rate <= 1e-5:
        learning_rate /= 3
    return learning_rate


def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean), name='get_epsilon')
    offset = tf.math.multiply(epsilon, z_log_var, name='offset')
    return z_mean + offset


class KLLoss:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, z, mean, logvar):
        loss = 1 + 2 * logvar - tf.math.square(mean) - tf.math.exp(2 * logvar)
        return - self.factor * .5 * tf.math.reduce_mean(loss)


class Encoder:
    def __init__(self):
        pass

    def __call__(self, x_input):
        x = ConvBlock(n_filters=32)(x_input)
        x = ConvBlock(n_filters=32)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=64)(x)
        x = ConvBlock(n_filters=64)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=128)(x)
        x = ConvBlock(n_filters=128)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=256)(x)
        x = ConvBlock(n_filters=256)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=256)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=256)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        x = ConvBlock(n_filters=512)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
        return tf.keras.layers.Flatten(name='out_encoder')(x)


class DecoderMask:
    def __init__(self):
        pass

    def __call__(self, z_latent, out_name, out_nc):
        x = tf.keras.layers.Reshape((1, 1, 512))(z_latent)
        x = ConvBlockTranspose(32, padding='same')(x)
        x = ConvBlock(32)(x)
        x = ConvBlockTranspose(32, padding='same')(x)
        x = ConvBlock(32)(x)
        x = ConvBlockTranspose(32, padding='valid')(x)
        x = ConvBlock(32)(x)
        x = ConvBlockTranspose(32, padding='same')(x)
        x = ConvBlock(32)(x)
        x = ConvBlockTranspose(32, padding='same')(x)
        x = ConvBlock(32)(x)
        x = ConvBlockTranspose(32, padding='same')(x)
        x = ConvBlock(32)(x)
        x = ConvBlockTranspose(32, padding='same')(x)
        x = ConvBlock(32)(x)
        return tf.keras.layers.Conv2D(out_nc, (3, 3), strides=(1, 1), activation='sigmoid',
                                      padding='same', name=out_name)(x)


class Decoder:
    def __init__(self):
        pass

    def __call__(self, z_latent, out_name, out_nc):
        x = tf.keras.layers.Reshape((1, 1, 512))(z_latent)
        x = ConvBlockTranspose(256, padding='same')(x)
        x = ConvBlock(256)(x)
        x = ConvBlockTranspose(256, padding='same')(x)
        x = ConvBlock(256)(x)
        x = ConvBlockTranspose(256, padding='valid')(x)
        x = ConvBlock(256)(x)
        x = ConvBlockTranspose(128, padding='same')(x)
        x = ConvBlock(128)(x)
        x = ConvBlockTranspose(64, padding='same')(x)
        x = ConvBlock(64)(x)
        x = ConvBlockTranspose(32, padding='same')(x)
        x = ConvBlock(32)(x)
        x = ConvBlockTranspose(32, padding='same')(x)
        x = ConvBlock(32)(x)
        return tf.keras.layers.Conv2D(out_nc, (3, 3), strides=(1, 1), activation='sigmoid',
                                      padding='same', name=out_name)(x)


class VAE2train:
    def __init__(self, factor_kl=1e-3):
        self.factor_kl = factor_kl
        self._build_model()

    def _build_model(self):
        tf.keras.backend.clear_session()
        self.in_image = tf.keras.layers.Input(shape=(144, 144, 3), name='in_image')
        self.in_mask = tf.keras.layers.Input(shape=(144, 144, 1), name='in_mask')

        out_encoder = Encoder()(self.in_image)

        x = tf.keras.layers.Dense(512)(out_encoder)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('elu', name='out_latent_1')(x)

        self.z_mean = tf.keras.layers.Dense(512, name='z_mean')(x)
        self.z_log_var = tf.keras.layers.Dense(512, name='z_logvar')(x)
        self.z_latent = tf.keras.layers.Lambda(sampling, output_shape=(512,),
                                               name='z_sampling')([self.z_mean, self.z_log_var])

        out_image_pre = Decoder()(self.z_latent, 'out_image_pre', 3)
        self.out_mask = DecoderMask()(self.z_latent, 'out_mask', 1)

        # Tidy the image to use only the face regions of the estimated output and join with the original background
        x = tf.keras.layers.Multiply()([out_image_pre, self.in_mask])
        x_bg = tf.keras.layers.Multiply()([self.in_image, 1. - self.in_mask])
        self.out_image = tf.keras.layers.Add(name='out_image')([x, x_bg])

        self.model = tf.keras.models.Model([self.in_image, self.in_mask],
                                           [self.out_image, self.out_mask, out_image_pre])

    def add_losses(self, add_KL=True, add_SSIM=True, add_MAE=True, add_BCE=True, add_MSE=False):
        self.checkpoint_fname = 'weights.{epoch:03d}-{loss:.2f}-{val_loss:.2f}'

        if add_KL:
            loss_KL = KLLoss(factor=self.factor_kl)(self.z_latent, self.z_mean, self.z_log_var)
            self.model.add_loss(loss_KL)
            self.model.add_metric(loss_KL, name='kl', aggregation='mean')
            self.checkpoint_fname += '-{val_kl:.2f}'

        if add_SSIM:
            loss_face_ssim = SSIMLoss()(self.in_image, self.out_image)
            self.model.add_loss(loss_face_ssim)
            self.model.add_metric(loss_face_ssim, name='ssim', aggregation='mean')
            self.checkpoint_fname += '-{val_ssim:.2f}'

        if add_MAE:
            loss_face_mae = 4 * tf.keras.losses.MeanAbsoluteError()(self.in_image, self.out_image)
            self.model.add_loss(loss_face_mae)
            self.model.add_metric(loss_face_mae, name='mae', aggregation='mean')
            self.checkpoint_fname += '-{val_mae:.2f}'

        if add_MSE:
            loss_face_mse = 100 * tf.keras.losses.MeanSquaredError()(self.in_image, self.out_image)
            self.model.add_loss(loss_face_mse)
            self.model.add_metric(loss_face_mse, name='mse', aggregation='mean')
            self.checkpoint_fname += '-{val_mse:.2f}'

        if add_BCE:
            loss_mask_bce = BCELoss()(self.in_mask, self.out_mask)
            self.model.add_loss(loss_mask_bce)
            self.model.add_metric(loss_mask_bce, name='bce', aggregation='mean')
            self.checkpoint_fname += '-{val_bce:.2f}'

        self.checkpoint_fname += '.h5'

    def setup_checkpoint(self, ckptdir=None, save_best_only=True):
        if ckptdir is None:
            self.ckptdir = os.path.join(self.traindir, 'ckpt')
        else:
            self.ckptdir = ckptdir

        if not os.path.exists(self.ckptdir):
            os.makedirs(self.ckptdir)

        self.checkpoint = ModelCheckpoint(os.path.join(self.ckptdir, self.checkpoint_fname),
                                          save_weights_only=True, save_best_only=save_best_only,
                                          save_format="tf", monitor='val_loss', verbose=1,
                                          mode='min')

    def setup_indir(self, traindir=None):
        if traindir is None:
            self.traindir = os.path.join('traindir', datetime.datetime.now().stime("%Y%m%d-%H%M%S"))
        else:
            self.traindir = traindir

    def setup_logdir(self):
        self.logdir = os.path.join(self.traindir, 'logs')
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir, exist_ok=True)

    def train(self, gen_in, gen_val, traindir=None, steps_per_epoch=4000, validation_steps=700, epochs=100,
              initial_epoch=0, save_best_only=True):
        self.setup_indir(traindir)
        self.setup_logdir()
        self.setup_checkpoint(save_best_only=save_best_only)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(self.logdir, histogram_freq=1)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1e-3))
        self.model.fit(gen_in,
                       validation_data=gen_val,
                       verbose=1,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=validation_steps,
                       epochs=epochs,
                       initial_epoch=initial_epoch,
                       callbacks=[self.checkpoint, tensorboard_callback])

    def train_checkup(self, gen_in, ckptdir='.', steps_per_epoch=10, epochs=2):
        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir, exist_ok=True)

        self.setup_checkpoint(ckptdir=ckptdir)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1e-3))
        self.model.fit(gen_in,
                       validation_data=gen_in,
                       verbose=1,
                       callbacks=[self.checkpoint],
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=2,
                       epochs=epochs)


class VAENoMask(VAE2train):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_model(self):
        tf.keras.backend.clear_session()
        self.in_image = tf.keras.layers.Input(shape=(144, 144, 3), name='in_image')

        self.out_encoder = Encoder()(self.in_image)

        x = tf.keras.layers.Dense(512)(self.out_encoder)
        # x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('elu', name='out_latent_1')(x)

        self.z_mean = tf.keras.layers.Dense(512, name='z_mean')(x)
        self.z_log_var = tf.keras.layers.Dense(512, name='z_logvar')(x)
        self.z_latent = tf.keras.layers.Lambda(sampling, output_shape=(512,),
                                               name='z_sampling')([self.z_mean, self.z_log_var])

        self.out_image = Decoder()(self.z_latent, 'out_image', 3)

        self.model = tf.keras.models.Model(self.in_image, self.out_image)


if __name__ == '__main__':
    celeba_dir = os.environ['celeba']
    list_eval_partition = pd.read_csv(os.path.join(celeba_dir, 'list_eval_partition.txt'), sep='\s+', header=None)
    list_eval_partition.columns = ['bname', 'set_id']
    list_eval_partition['path'] = list_eval_partition['bname'].apply(lambda x: os.path.join(celeba_dir, 'imgs', x))

    fpaths_in = list_eval_partition.query('set_id == 0')['path'].values.tolist()
    fpaths_val = list_eval_partition.query('set_id == 1')['path'].values.tolist()
    fpaths_test = list_eval_partition.query('set_id == 2')['path'].values.tolist()

    np.random.seed(5)
    for fset in [fpaths_in, fpaths_val, fpaths_test]:
        np.random.shuffle(fset)

    gen_in = ut.InputGen(impaths=fpaths_in, loadsize_factor=2)
    gen_val = ut.InputGen(impaths=fpaths_val, loadsize_factor=2)
    vae = VAE2train()
    vae.train(gen_in, gen_val)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        #print(z_mean.shape)
        batch = tf.shape(z_mean)[0]
        dim1 = tf.shape(z_mean)[1]
        dim2 = tf.shape(z_mean)[2]
        ch = tf.shape(z_mean)[3]
        sh = (batch, dim1, dim2, ch)
        #print(sh)
        epsilon =  tf.random.normal(shape=sh)
        #epsilon = np.random.multivariate_normal(np.zeros(15), np.identity(15), size=(batch,dim1))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class SamplingDense(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class FCVAE(tf.keras.Model):
    def __init__(self, lat_size=500, input_dim=(48,48,4), **kwargs):
        super(FCVAE, self).__init__(**kwargs)

        self.img_rows = input_dim[0]
        self.img_cols = input_dim[1]
        self.channels = input_dim[2]
        self.kernel_size = 5
        self.ff = 4
        self.latent_size = lat_size
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = int(self.img_rows/(2**2))
        self.latent_dim_shape = (self.latent_dim, self.latent_dim, self.latent_size)
        self.epochs_drop = 20

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    # learning rate schedule
    def step_decay(self,epoch):
        self.initial_lrate = tf.keras.backend.eval(self.optimizer.lr)
        drop = 0.8
        if (1+epoch)%self.epochs_drop == 0:
            #lrate = self.initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            lrate=self.initial_lrate*drop
        else:
            lrate=self.initial_lrate
        return lrate
    
    def build_encoder(self):

        encoder_inputs = tf.keras.Input(shape=(None,None,self.channels))
        x = tf.keras.layers.Conv2D(8*self.ff, 5, activation="relu", strides=1, padding="same")(encoder_inputs)
        x = tf.keras.layers.Conv2D(8*self.ff, 5, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.Conv2D(16*self.ff, 5, activation="relu", strides=2, padding="same")(x)
        
        # Regularization techniques
        #x = tf.keras.layers.Dropout(0.1)(x)  # Dropout regularization
        #x = tf.keras.layers.BatchNormalization()(x)  # Batch normalization

        x = tf.keras.layers.Conv2D(16*self.ff, 3, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.Conv2D(32*self.ff, 3, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.Conv2D(32*self.ff, 3, activation="relu", strides=2, padding="same")(x)
        
        # Regularization techniques
        #x = tf.keras.layers.Dropout(0.1)(x)  # Dropout regularization
        #x = tf.keras.layers.BatchNormalization()(x)  # Batch normalization

        z_mean = tf.keras.layers.Conv2D(self.latent_size, 3, strides=1, padding='same', name="z_mean")(x)
        z_log_var = tf.keras.layers.Conv2D(self.latent_size, 3, strides=1, padding='same', name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        return encoder

    def build_decoder(self):

        latent_inputs = tf.keras.Input(shape=(None, None, self.latent_size))
        x = tf.keras.layers.Conv2DTranspose(32*self.ff, 3, activation="relu", strides=2, padding="same")(latent_inputs)
        x = tf.keras.layers.Conv2DTranspose(32*self.ff, 3, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(16*self.ff, 3, activation="relu", strides=1, padding="same")(x)
        
        # Regularization techniques
        #x = tf.keras.layers.Dropout(0.1)(x)  # Dropout regularization
        #x = tf.keras.layers.BatchNormalization()(x)  # Batch normalization
        
        x = tf.keras.layers.Conv2DTranspose(16*self.ff, 5, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(8*self.ff, 5,activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(8*self.ff, 5, activation="relu", strides=1, padding="same")(x)
        
        # Regularization techniques
        #x = tf.keras.layers.Dropout(0.9)(x)  # Dropout regularization
        #x = tf.keras.layers.BatchNormalization()(x)  # Batch normalization
        decoder_outputs = tf.keras.layers.Conv2D(self.channels, 5, activation="sigmoid", padding="same")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        return decoder
    
    def decode(self, noise):
        return tf.argmax(self.decoder(noise), axis=-1)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        clear_output(wait=False)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=[1,2,3]))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class VAE(tf.keras.Model):
    def __init__(self, lat_size=500, input_dim=(48,48,4), **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.img_rows = input_dim[0]
        self.img_cols = input_dim[1]
        self.channels = input_dim[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = lat_size
        self.latent_dim_shape = (self.latent_dim)

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    # learning rate schedule
    def step_decay(self,epoch):
        self.initial_lrate = tf.keras.backend.eval(self.model.optimizer.lr)
        drop = 0.8
        if (1+epoch)%self.epochs_drop == 0:
            #lrate = self.initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            lrate=self.initial_lrate*drop
        else:
            lrate=self.initial_lrate

    def build_encoder(self):

        encoder_inputs = tf.keras.Input(shape=self.img_shape)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        #x = tf.keras.layers.Dropout(0.1)(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = SamplingDense()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        return encoder

    def build_decoder(self):

        latent_inputs = tf.keras.Input(shape=self.latent_dim_shape,)
        x = tf.keras.layers.Dense(1024, activation="relu")(latent_inputs)
        #x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.latent_dim*16, activation="relu")(x)
        x = tf.keras.layers.Reshape((12, 12, 16))(x)
        x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(self.channels, 2, activation="sigmoid", padding="same")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        return decoder
    
    def decode(self, noise):
        return tf.argmax(self.decoder(noise), axis=-1)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        clear_output(wait=False)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
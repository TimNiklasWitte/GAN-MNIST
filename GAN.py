import tensorflow as tf

from Generator import *
from Discriminator import *

class GAN(tf.keras.Model):

    def __init__(self):
        super(GAN, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.bce_loss = tf.keras.losses.BinaryCrossentropy()


    @tf.function
    def train_step(self, img_real):

        batch_size = img_real.shape[0]
        noise = tf.random.uniform(minval=-1, maxval=1, shape=(batch_size, self.generator.noise_dim))

        ones = tf.ones(shape=(batch_size,))
        zeros = tf.zeros(shape=(batch_size,))
        with tf.GradientTape(persistent=True) as tape:

            img_fake = self.generator(noise, training=True)

            rating_fake = self.discriminator(img_fake, training=True)
            rating_real = self.discriminator(img_real, training=True)

            # generator
            generator_loss = self.bce_loss(ones, rating_fake)

            # discriminator
            discriminator_fake_loss = self.bce_loss(zeros, rating_fake)
            discriminator_real_loss = self.bce_loss(ones, rating_real)
            discriminator_loss = discriminator_fake_loss + discriminator_real_loss


        # Update generator
        gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        

        # Update discriminator
        gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))


        #
        # Update metrices
        #

        # Generator
        self.generator.metric_loss.update_state(generator_loss)

        # Discriminator

        # Loss
        self.discriminator.metric_loss.update_state(discriminator_loss)
        self.discriminator.metric_fake_loss.update_state(discriminator_fake_loss)
        self.discriminator.metric_real_loss.update_state(discriminator_real_loss)

        # Accuracy
        classified_fake = tf.math.round(rating_fake)
        classified_real = tf.math.round(rating_real)

        self.discriminator.metric_fake_accuracy.update_state(zeros, classified_fake)
        self.discriminator.metric_real_accuracy.update_state(ones, classified_real)
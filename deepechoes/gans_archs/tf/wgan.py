import tensorflow as tf
import time
from matplotlib import pyplot as plt
from gans_archs.tf.base import BaseGAN

class WGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def discriminator_loss(self, real_output, fake_output, gradient_penalty):
        # Loss combines WGAN loss and gradient penalty from the WGAN-GP paper
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + self.lambda_gp * gradient_penalty

    def generator_loss(self, fake_output):
        # Mmaximize discriminator's prediction on fake outputs
        return -tf.reduce_mean(fake_output)

    def gradient_penalty(self, real_images, fake_images):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Get the interpolated image
        alpha = tf.random.uniform([len(real_images), 1, 1, 1], 0., 1.)
        interpolated = real_images * alpha + fake_images * (1 - alpha)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp
    
    @tf.function
    def train_step(self, images, noise_dim):
        noise = tf.random.normal([len(images), noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gp = self.gradient_penalty(images, generated_images)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output, gp)

        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)
        self._disc_accuracy.update_state(real_labels, real_output)
        self._disc_accuracy.update_state(fake_labels, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Storing the losses to get the epoch mean later
        self._gen_loss(gen_loss)
        self._disc_loss(disc_loss)

        return gen_loss, disc_loss
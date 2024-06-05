import tensorflow as tf
from gans_archs.tf.base import BaseGAN

class WGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
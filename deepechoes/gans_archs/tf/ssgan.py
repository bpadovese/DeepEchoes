import tensorflow as tf
from deepechoes.gans_archs.tf.base import BaseGAN
from deepechoes.utils.image_transforms import rotate_images_and_labels

class SSGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disc_weight_rotation_loss = 1.0
        self.gen_weight_rotation_loss = 1.0

    
    def rotation_loss(true_rotations, rotation_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(true_rotations, rotation_pred, from_logits=True)
        return tf.reduce_mean(loss)

    @tf.function
    def train_step(self, images, noise_dim):
        noise = tf.random.normal([len(images), noise_dim])

        # Rotate images and generate labels for rotations. Every image will generate 4. So a 32,4,4,1 spectrogram will become 32*4,4,4,1
        rotated_images_real, rotation_labels_real = rotate_images_and_labels(images)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            # rotated generated images
            rotated_generated_images, rotation_labels_fake = rotate_images_and_labels(generated_images)
            
            # Discriminator output for real images (only upright images are used for fake vs. real classification)
            # The discriminator has two heads, and returns two outputs, one for the task of real vs fake classfication, and another for the rotation classficaiton
            real_output , _ = self.discriminator(images, training=True)
            fake_output, _ = self.discriminator(generated_images, training=True)

            # Discriminator outputs for rotated images
            _, rotation_logits_real = self.discriminator(rotated_images_real, training=True)
            _, rotation_logits_fake = self.discriminator(rotated_generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            rotation_loss_real = self.rotation_loss(rotation_labels_real, rotation_logits_real)
            rotation_loss_fake = self.rotation_loss(rotation_labels_fake, rotation_logits_fake)

            # Total discriminator loss
            disc_loss_total = disc_loss + self.disc_weight_rotation_loss * rotation_loss_real
            gen_total_loss = gen_loss + self.gen_weight_rotation_loss * rotation_loss_fake

        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)
        self._disc_accuracy.update_state(real_labels, real_output)
        self._disc_accuracy.update_state(fake_labels, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss_total, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Storing the losses to get the epoch mean later
        self._gen_loss(gen_loss)
        self._disc_loss(disc_loss_total)

        return gen_loss, disc_loss_total
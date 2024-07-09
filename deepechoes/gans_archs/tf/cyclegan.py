import tensorflow as tf
from pathlib import Path
from deepechoes.gans_archs.tf.base import BaseGAN

class CycleGAN(BaseGAN):
    def __init__(self, generator_g, generator_f, discriminator_x, discriminator_y, gen_g_optimizer, gen_f_optimizer, disc_x_optimizer, disc_y_optimizer, loss_fn='cycle_gan'):
        super().__init__(generator_g, discriminator_x, gen_g_optimizer, disc_x_optimizer, loss_fn)
        self.generator_f = generator_f
        self.discriminator_y = discriminator_y
        self.gen_f_optimizer = gen_f_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        
        self.checkpoint = tf.train.Checkpoint(generator_g_optimizer=self.gen_optimizer,
                                              generator_f_optimizer=self.gen_f_optimizer,
                                              discriminator_x_optimizer=self.disc_optimizer,
                                              discriminator_y_optimizer=self.disc_y_optimizer,
                                              generator_g=self.generator,
                                              generator_f=self.generator_f,
                                              discriminator_x=self.discriminator,
                                              discriminator_y=self.discriminator_y)
        
    def set_loss_fn(self, loss):
        self.cycle_loss_weight = 10.0
        self.identity_loss_weight = 5.0
        super().set_loss_fn(loss)
        
    @staticmethod
    def cycle_consistency_loss(real_image, cycled_image, loss_fn):
        return tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    @staticmethod
    def identity_loss(real_image, same_image, loss_fn):
        return tf.reduce_mean(tf.abs(real_image - same_image))
    
    @tf.function
    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X to Y
            fake_y = self.generator(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)
            
            # Generator F translates Y to X
            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator(fake_x, training=True)
            
            # Identity mapping
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator(real_y, training=True)
            
            # Discriminator output
            disc_real_x = self.discriminator(real_x, training=True)
            disc_fake_x = self.discriminator(fake_x, training=True)
            
            disc_real_y = self.discriminator_y(real_y, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)
            
            # Calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)
            
            cycle_loss_g = self.cycle_consistency_loss(real_x, cycled_x, self.generator_loss)
            cycle_loss_f = self.cycle_consistency_loss(real_y, cycled_y, self.generator_loss)
            
            total_cycle_loss = cycle_loss_g + cycle_loss_f
            
            identity_loss_g = self.identity_loss(real_y, same_y, self.generator_loss)
            identity_loss_f = self.identity_loss(real_x, same_x, self.generator_loss)
            
            total_gen_g_loss = gen_g_loss + total_cycle_loss * self.cycle_loss_weight + identity_loss_g * self.identity_loss_weight
            total_gen_f_loss = gen_f_loss + total_cycle_loss * self.cycle_loss_weight + identity_loss_f * self.identity_loss_weight
            
            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
        
        # Calculate the gradients for generators and discriminators
        gradients_of_generator_g = tape.gradient(total_gen_g_loss, self.generator.trainable_variables)
        gradients_of_generator_f = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
        
        gradients_of_discriminator_x = tape.gradient(disc_x_loss, self.discriminator.trainable_variables)
        gradients_of_discriminator_y = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)
        
        # Apply the gradients to the optimizers
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator_g, self.generator.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(gradients_of_generator_f, self.generator_f.trainable_variables))
        
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator_x, self.discriminator.trainable_variables))
        self.disc_y_optimizer.apply_gradients(zip(gradients_of_discriminator_y, self.discriminator_y.trainable_variables))
        
        return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss

import time
import tensorflow as tf
from deepechoes.dev_utils.image_transforms import unscale_data
from pathlib import Path
from matplotlib import pyplot as plt

class BaseGAN:
    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer, loss_fn="bce"):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self._generated_image_dir = None
        self._checkpoint_dir = None
        self._log_dir = None

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                    discriminator_optimizer=self.disc_optimizer,
                                    generator=self.generator,
                                    discriminator=self.discriminator)

        self._disc_loss = tf.keras.metrics.Mean(name='disc_loss')
        self._gen_loss = tf.keras.metrics.Mean(name='gen_loss')
        self._disc_accuracy = tf.keras.metrics.BinaryAccuracy(name='discriminator_accuracy')

        self.set_loss_fn(loss_fn)

    @property
    def generated_image_dir(self):
        return self._generated_image_dir

    @generated_image_dir.setter
    def generated_image_dir(self, generated_image_dir):
        """ Defines the directory where tensorflow checkpoint files can be stored

            Args:
                generated_image_dir: str or Path
                    Path to the directory

        """
        if isinstance(generated_image_dir, Path):
            self._generated_image_dir = generated_image_dir
        else:
            self._generated_image_dir = Path(generated_image_dir)

        self.generated_image_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, checkpoint_dir):
        """ Defines the directory where tensorflow checkpoint files can be stored

            Args:
                checkpoint_dir: str or Path
                    Path to the directory

        """
        if isinstance(checkpoint_dir, Path):
            self._checkpoint_dir = checkpoint_dir
        else:
            self._checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup the checkpoint manager now that we know where the checkpoints will be stored
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, directory=str(self.checkpoint_dir), max_to_keep=4)

    @property
    def log_dir(self):
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        """ Defines the directory where tensorflow checkpoint files can be stored

            Args:
                log_dir: str or Path
                    Path to the directory

        """
        if isinstance(log_dir, Path):
            self._log_dir = log_dir
        else:
            self._log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def set_loss_fn(self, loss):
        match loss:
            case 'bce':
                self.generator_loss = self.bce_generator_loss
                self.discriminator_loss = self.bce_discriminator_loss
            case 'hinge':
                self.generator_loss = self.shared_generator_loss
                self.discriminator_loss = self.hinge_discriminator_loss
            case 'wgan_gp':
                self.generator_loss = self.shared_generator_loss
                self.discriminator_loss = self.wgan_gp_discriminator_loss
            case _:
                raise ValueError("Unsupported loss type")
    
    def wgan_gp_discriminator_loss(self, real_output, fake_output, gradient_penalty, lambda_gp=10):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + lambda_gp * gradient_penalty

    
    def gradient_penalty(self, real_images, fake_images):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = tf.random.uniform([len(real_images), 1, 1, 1], 0., 1.)
        interpolated = real_images * alpha + fake_images * (1 - alpha)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True) 

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
    
    def shared_generator_loss(fake_output):
        return -tf.reduce_mean(fake_output)
    
    @staticmethod
    def bce_generator_loss(fake_output):
        return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True)

    @staticmethod
    def bce_discriminator_loss(real_output, fake_output):
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
        fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
        return real_loss + fake_loss

    @staticmethod
    def hinge_discriminator_loss(real_output, fake_output):
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
        return real_loss + fake_loss

    def train_step(self, input):
        raise NotImplementedError

    def train_loop(self, batch_generator, epochs, checkpoint_freq=5, noise_vector=None):
        generator_losses = []
        discriminator_losses = []
        discriminator_accuracies = []
        
        noise_dim = 100
        num_examples_to_generate = 16
        noise = noise_vector

        for epoch in range(epochs):
            start = time.time()
            self._gen_loss.reset_states()
            self._disc_loss.reset_states()
            self._disc_accuracy.reset_states()

            for _ in range(batch_generator.n_batches):
                # Fetch a batch of data
                train_X, _ = next(batch_generator)
                self.train_step(train_X, noise_dim)
            
            avg_gen_loss = self._gen_loss.result().numpy()
            avg_disc_loss = self._disc_loss.result().numpy()
            avg_disc_accuracy = self._disc_accuracy.result().numpy()

            print(f'Epoch {epoch + 1}, Avg Gen Loss={avg_gen_loss:.4f}, Avg Disc Loss={avg_disc_loss:.4f}, Disc Accuracy={avg_disc_accuracy:.4f}')
            
            # Append the average losses for plotting later
            generator_losses.append(avg_gen_loss)
            discriminator_losses.append(avg_disc_loss)
            discriminator_accuracies.append(avg_disc_accuracy)

            if noise_vector is None:
                noise = tf.random.normal([num_examples_to_generate, noise_dim]) 
                
            # Produce images
            self.generate_and_plot_images(self.generator, epoch + 1, noise)
            
            if (epoch + 1) % checkpoint_freq == 0:
                self.ckpt_manager.save(checkpoint_number=epoch + 1)

            print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

        # Generate after the final epoch
        self.generate_and_plot_images(self.generator, epochs, noise)
        
        if self.log_dir is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

            # Loss plot
            ax1.plot(range(1, epochs + 1), generator_losses, label="Generator Loss")
            ax1.plot(range(1, epochs + 1), discriminator_losses, label="Discriminator Loss")
            ax1.set_title("Average Loss Over Epochs")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()

            # Accuracy plot
            ax2.plot(range(1, epochs + 1), discriminator_accuracies, color='orange', label="Discriminator Accuracy")
            ax2.set_title("Discriminator Accuracy Over Epochs")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(self.log_dir / 'training_metrics.png')

    def save(self, output_dir, save_discriminator=True):
        if isinstance(output_dir, Path):
            output_dir = output_dir
        else:
            output_dir = Path(output_dir)
            # Save the generator model
        self.generator.save(output_dir / 'generator_model')
        
        # Optionally save the discriminator model
        if save_discriminator:
            self.discriminator.save(output_dir / 'discriminator_model')

    def generate_new(self, num_samples=10):
        """ Generate new spectrograms using the generator """
        # Assumes generator expects a random noise vector as input
        noise_dim = 100  
        random_noise = tf.random.normal([num_samples, noise_dim])
        
        generated_images = self.generator(random_noise, training=False)
        return generated_images

    def generate_and_plot_images(self, model, epoch, input):
        predictions = model(input, training=False)
        fig, axs = plt.subplots(4, 4, figsize=(34, 28))
        plt.subplots_adjust(wspace=0, hspace=0)  # Adjust as needed
        for i in range(predictions.shape[0]):
            ax = axs[i // 4, i % 4]
            mel_spectrogram = unscale_data(predictions[i, :, :, 0].numpy())
            ax.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.generated_image_dir / 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.close(fig)

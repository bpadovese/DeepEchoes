import tensorflow as tf
from deepechoes.constants import OUTPUT_CHANNELS, INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH

def phase_shuffle(x, max_phase_shift):
    # x is a 4D tensor with shape (batch_size, height, width, channels)
    batch_size, height, width, channels = x.shape
    phase_shift = tf.random.uniform(shape=(), minval=-max_phase_shift, maxval=max_phase_shift + 1, dtype=tf.int32)
    
    return tf.roll(x, shift=phase_shift, axis=2)  # Shift along width

def dcgans_discriminator_block(filters, kernel_size, strides=(2, 2), apply_norm=True, norm_type='instance', phase_shift=0):
    model = tf.keras.Sequential()
    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False))
    
    if apply_norm:
        if norm_type == 'batch':
            model.add(tf.keras.layers.BatchNormalization())
        elif norm_type == 'layer':
            model.add(tf.keras.layers.LayerNormalization())
        elif norm_type == 'instance':
            model.add(tf.keras.layers.GroupNormalization(groups=-1, axis=-1))
    
    model.add(tf.keras.layers.LeakyReLU())

    if phase_shift != 0:
        model.add(tf.keras.layers.Lambda(lambda x: phase_shuffle(x, phase_shift)))
    
    return model

def DcgansDiscriminator(apply_norm=True, norm_type='batch', phase_shift=0):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS)))  # 128x128x1
    
    # Downsample the input
    model.add(dcgans_discriminator_block(64, 5, strides=2, apply_norm=False, phase_shift=phase_shift))  # 64x64x64, no batchnorm on first layer
    model.add(dcgans_discriminator_block(128, 5, strides=2, apply_norm=apply_norm, norm_type=norm_type, phase_shift=phase_shift))  # 32x32x128
    model.add(dcgans_discriminator_block(256, 5, strides=2, apply_norm=apply_norm, norm_type=norm_type, phase_shift=phase_shift))  # 16x16x256
    model.add(dcgans_discriminator_block(512, 5, strides=2, apply_norm=apply_norm, norm_type=norm_type, phase_shift=phase_shift)) # 8x8x512

    # Output layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    
    return model

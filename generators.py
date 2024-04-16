import tensorflow as tf
from constants import OUTPUT_CHANNELS, INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH

# Encoder
def downsample_block(filters, kernal_size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, kernal_size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

# Decoder
def upsample_block(filters, kernal_size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, kernal_size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def UnetGenerator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS])

    down_stack = [
        downsample_block(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample_block(128, 4),  # (batch_size, 64, 64, 128)
        downsample_block(256, 4),  # (batch_size, 32, 32, 256)
        downsample_block(512, 4),  # (batch_size, 16, 16, 512)
        downsample_block(512, 4),  # (batch_size, 8, 8, 512)
        downsample_block(512, 4),  # (batch_size, 4, 4, 512)
        downsample_block(512, 4),  # (batch_size, 2, 2, 512)
        # downsample_block(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        # upsample_block(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample_block(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample_block(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample_block(512, 4),  # (batch_size, 16, 16, 1024)
        upsample_block(256, 4),  # (batch_size, 32, 32, 512)
        upsample_block(128, 4),  # (batch_size, 64, 64, 256)
        upsample_block(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 1)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def dcgans_block(filters, kernal_size, strides=(2,2), apply_norm=True, norm_type="instance"):
    result = tf.keras.Sequential()
    initializer = tf.random_normal_initializer(0., 0.02)

    result.add(tf.keras.layers.Conv2DTranspose(filters, kernal_size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_norm:
        if norm_type == 'batch':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type == 'layer':
            result.add(tf.keras.layers.LayerNormalization())
        elif norm_type == 'instance':
            result.add(tf.keras.layers.GroupNormalization(groups=-1, axis=-1))
    result.add(tf.keras.layers.ReLU())

    return result


def DcgansGenerator(input_shape=100, apply_norm=True, norm_type='batch'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8*8*512, use_bias=False, input_shape=(input_shape,)))
    if apply_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Reshape((8, 8, 512))) # reshaping to a 8x8 feature map

    stack = [
        dcgans_block(512, 5, strides=1, apply_norm=apply_norm, norm_type=norm_type),  # Keeping size
        dcgans_block(256, 5, strides=2, apply_norm=apply_norm, norm_type=norm_type), # 16x16x256
        dcgans_block(128, 5, strides=2, apply_norm=apply_norm, norm_type=norm_type), # 32x32x128
        dcgans_block(64, 5, strides=2, apply_norm=apply_norm, norm_type=norm_type), # 64x64x64
    ]

    for block in stack:
        model.add(block)
    
    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer, activation='tanh'))

    return model

class ResNetBlock(tf.keras.Model):
    """ Residual block for ResNet architectures.

        Args: 
            filters: int
                The number of filters in the block
            strides: int
                Strides used in convolutional layers within the block
            kernel: (int,int)
                Kernel used in convolutional layers within the block
            residual_path: bool
                Whether or not the block will contain a residual path
            batch_norm_momentum: float between 0 and 1
                Momentum for the moving average of the batch normalization layers.
                The default value is 0.99.
                For an explanation of how the momentum affects the batch normalisation operation,
                see <https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization>

        Returns:
            A ResNetBlock object. The block itself is a tensorflow model and can be used as such.
    """
    def __init__(self, filters, strides=1, kernel=(3,3), residual_path=False, batch_norm_momentum=0.99):
        super(self).__init__()

        self.filters = filters
        self.strides = strides
        self.kernel  = kernel
        self.residual_path = residual_path

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.batch_norm_1 = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum)

        self.conv_2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, strides=1,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.batch_norm_2 = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum)

        if residual_path == True:
            self.conv_down = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1,1), strides=self.strides,
                                                padding="same", use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.02))

            self.batch_norm_down = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum)

    def call(self,inputs, training=None):
        """Calls the model on new inputs.

        In this case call just reapplies all ops in the graph to the new inputs (e.g. build a new computational graph from the provided inputs).

        Args:
            inputs: Tensor or list of tensors
                A tensor or list of tensors
            
            training: Bool
                Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode.

        Returns:
                A tensor if there is a single output, or a list of tensors if there are more than one outputs.
        """
        residual = inputs

        x = self.batch_norm_1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.batch_norm_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_2(x)

        if self.residual_path:
            residual = self.batch_norm_down(inputs, training=training)
            residual = tf.nn.relu(residual)
            x = self.conv_down(residual)

        x = x + residual
        return x
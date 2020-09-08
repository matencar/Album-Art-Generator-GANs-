import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, GaussianNoise, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from tensorflow.layers import InstanceNormalization

class ADAin(Layer):
    def __init__(self):
        super(MyDenseLayer, self).__init__()

    # def build(self):

    def call(self, y_s, y_b, x):
        return ADAin(y_s, y_b, x)

    def ADAin(y_s, y_b, x):
        """
        Performs the ADAin calculation.
        y_s = scale vector, size (1, n)
        y_b = bias vector, size (1, n)
        x = convolutional output
        """

def make_generator(num_channels):
    """
    The model for the generator network is defined here. 
    """
    # instantiate model
    model = Sequential()

    # starts with shape (z_dim * 4 * 4) random input
    # add noise
    model.add(GaussianNoise(stddev=1), input_size=(4, 4, z_dim))
    # do ADAin with vector in W space (from mapping net)
    model.add(ADAin())
    # do Conv 3x3
    model.add(Conv2DTranspose(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    # add noise
    model.add(GaussianNoise(stddev=1))
    # do ADAin with vector in W space (from mapping net)
    model.add(ADAin())

    # do upsampling
    # add noise
    model.add(GaussianNoise(stddev=1))
    # do ADAin with vector in W space (from mapping net)
    model.add(ADAin())
    # do Conv 3x3
    model.add(Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    # add noise
    model.add(GaussianNoise(stddev=1))
    # do ADAin with vector in W space (from mapping net)
    model.add(ADAin())

    # continue...

    return model

class Generator_Model(tf.keras.Model):
	def __init__(self, num_channels, z_dim):
		"""
		The model for the generator network is defined here. 
		"""
		super(Generator_Model, self).__init__()

        self.deconv1 = Conv2DTranspose(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.deconv2 = Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.deconv3 = Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.deconv4 = Conv2DTranspose(num_channels, (3, 3), strides=(2, 2), padding='same', use_bias=False)

        self.noise = GaussianNoise(stddev=1)
		

	@tf.function
	def call(self, scale, bias, z_dim):
		"""
		Executes the generator model on the random noise vectors.

		:param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

		:return: prescaled generated images, shape=[batch_size, height, width, channel]
		"""
        synthesized = tf.random.normal((4, 4, z_dim))
        synthesized = self.noise(synthesized)
        synthesized = self.ADAin(scale, bias, synthesized)
        synthesized = self.deconv1(synthesized)
        synthesized = self.noise(synthesized)
        synthesized = self.ADAin(scale, bias, synthesized)

        synthesized = tf.reshape(synthesized, ())
        synthesized = self.noise(synthesized)
        synthesized = self.ADAin(scale, bias, synthesized)
        synthesized = self.deconv2(synthesized)
        synthesized = self.noise(synthesized)
        synthesized = self.ADAin(scale, bias, synthesized)

        synthesized = tf.reshape(synthesized, ())
        synthesized = self.noise(synthesized)
        synthesized = self.ADAin(scale, bias, synthesized)
        synthesized = self.deconv2(synthesized)
        synthesized = self.noise(synthesized)
        synthesized = self.ADAin(scale, bias, synthesized)
        

		return 

	@tf.function
	def loss_function(self, disc_fake_output):
		"""
		Outputs the loss given the discriminator output on the generated images.

		:param disc_fake_output: the discrimator output on the generated images, shape=[batch_size,1]

		:return: loss, the cross entropy loss, scalar
		"""
		return cross_entropy(tf.ones_like(disc_fake_output), disc_fake_output)

    def ADAin(y_s, y_b, x):
        """
        Performs the ADAin calculation.
        y_s = scale vector, size (1, n)
        y_b = bias vector, size (1, n)
        x = convolutional output
        """
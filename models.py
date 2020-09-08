import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Concatenate, Embedding, Layer, GaussianNoise, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from tensorflow.nn import batch_normalization, leaky_relu

import numpy as np

from get_args import get_args
args = get_args()
z_dim = args.z_dim
batch_size = args.batch_size
num_channels = args.num_channels
num_genres = args.num_genres
num_gen_updates = args.num_gen_updates
out_dir = args.out_dir
mapping_dim = args.mapping_dim
image_side_len = args.image_side_len
n_genres = args.num_genres

num_convolutions = 4

def make_noise_scale_net():
	"""
	Analyzes a noise matrix to determine a scaling for the noise (scalar value)
	"""
	model = Sequential()

	initial_shape = 512

	model.add(Dense(initial_shape, use_bias=True, activation=leaky_relu))
	model.add(Dense(initial_shape // 2, use_bias=True, activation=leaky_relu))
	model.add(Dense(initial_shape // 4, use_bias=True, activation=leaky_relu))
	model.add(Dense(initial_shape // 8, use_bias=True, activation=leaky_relu))
	model.add(Dense(initial_shape // 16, use_bias=True, activation=leaky_relu))
	model.add(Dense(initial_shape // 32, use_bias=True, activation=leaky_relu))
	model.add(Dense(initial_shape // 64, use_bias=True, activation=leaky_relu))
	model.add(Dense(1, use_bias=True))

	return model

def make_affine_transform_net():
	"""
	Given a style vector of shape (z_dim), learns scale and bias channels for ADAin
	"""
	model = Sequential()

	upspace = num_channels * 4

	model.add(Dense(upspace, use_bias=True, input_shape=(z_dim,), activation=leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=leaky_relu))
	# model.add(Dense(2 * num_channels, use_bias=True))
	# model.add(Reshape((2, num_channels)))
	model.add(Dense(4 * num_channels, use_bias=True))
	model.add(Reshape((2, 2 * num_channels)))
	model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	return model
	# return ADAin_Model()

def make_mapping_net():
	"""
	Uses a random vector from latent space to learn styles of genres, outputs embeddings.
	Its output is fed into the affine transform layer.
	"""
	# instantiate model
	model = Sequential()

	# 8 layer MLP (f)
	model.add(Dense(mapping_dim, use_bias=True, input_shape=(z_dim,), activation=leaky_relu, name='Mapping1'))
	model.add(Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping2'))
	model.add(Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping3'))
	model.add(Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping4'))
	model.add(Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping5'))
	model.add(Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping6'))
	model.add(Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping7'))
	model.add(Dense(z_dim, use_bias=True, name='Mapping8'))

	return model

class ADAin(Layer):
	def __init__(self):
		super(ADAin, self).__init__()

	# def build(self):

	def call(self, x, y_s, y_b):
		return self.ADAin_calc(x, y_s, y_b)

	def expand(self, x, dims):
		for i in dims:
			x = tf.expand_dims(x, i)
		return x

	def ADAin_calc(self, x, y_s, y_b):
		"""
		Performs the ADAin calculation.
		y_s = scale vector, shape (batch_size, n)
		y_b = bias vector, shape (batch_size, n)
		x = convolutional output, shape (batch_size, width, height, out_channels)
		"""

		(mean, variance) = tf.nn.moments(x, (1, 2), keepdims=True)
		# result = tf.zeros_like(x)
		# result = np.zeros(x.shape)
		# channels = []
		# batch = []
		# for i in range(x.shape[0]): # for each batch
		# for j in range(x.shape[3]): # for each channel
			# channels.append((x[:, :, :, j] - self.expand_dimensions(mean, [1, 2]))/tf.sqrt(self.expand_dimensions(variance, [1, 2])) * tf.expand_dims(y_s[:][j], 1) + tf.expand_dims(y_b[:][j], 1))
		# batch.append(tf.stack(channels))
		# return tf.stack(channels)
		# return batch_normalization(x=x, mean=mean, variance=variance, offset=0, scale=1, variance_epsilon=0.00001)
		# return batch_normalization(x=x, mean=mean, variance=variance, offset=y_b, scale=y_s, variance_epsilon=0.00001)
		return ((x - mean)/tf.sqrt(variance)) * self.expand(y_s, [1, 2]) + self.expand(y_b, [1, 2])

class ScaledGaussianNoise(Layer):
	def __init__(self, stddev=1):
		super(ScaledGaussianNoise, self).__init__()
		self.stddev = stddev

	# def build(self):

	def call(self, x, scale=1):
		return x + scale * tf.random.normal(x.shape, stddev=self.stddev)

class Generator_Model(Model):
	def __init__(self):
		"""
		The model for the generator network is defined here. 
		"""
		super(Generator_Model, self).__init__()

		# side length image is reduced to by discriminator
		self.smallest_img_side_len = (image_side_len // (num_convolutions**2))

		# learned constant in StyleGAN
		self.learned_const = tf.constant(tf.random.normal((batch_size, self.smallest_img_side_len, self.smallest_img_side_len, num_channels)))

		# cGAN embeddings of genres
		self.embedding_size = 50
		self.embedding = Embedding(n_genres, self.embedding_size)
		self.embed_dense = Dense(self.smallest_img_side_len ** 2, activation=leaky_relu)
		self.embed_reshape = Reshape((self.smallest_img_side_len, self.smallest_img_side_len, 1))

		# cGAN prepping for random constant
		self.initial_dense = Dense(num_channels, activation=leaky_relu)
		self.initial_reshape = Reshape((self.smallest_img_side_len, self.smallest_img_side_len, num_channels))

		# StyleGAN layers
		self.deconv1 = Conv2DTranspose(num_channels // 2, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv2 = Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv3 = Conv2DTranspose(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False)

		self.finaldeconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False)

		# various layers
		self.noise = ScaledGaussianNoise(stddev=1)
		self.adain = ADAin()
		self.activation = LeakyReLU(0.2)
		self.merge = Concatenate()

	@tf.function
	def call(self, adain_net, w, genres, noise_scale=1):
		"""
		Executes the generator model on the random noise vectors.

		:param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

		:return: prescaled generated images, shape=[batch_size, height, width, channel]
		"""
		genres = tf.convert_to_tensor(genres)

		# in lieu of input, use a random constant as base

		# get embedding of genre label
		embed = self.embedding(genres)
		embed = self.embed_reshape(self.embed_dense(embed))
		out = self.initial_reshape(self.initial_dense(self.learned_const))
		# out = self.learned_const

		# merge embedding information and image information
		combined = self.merge([embed, out])
		# proceed with StyleGAN
		(scale, bias) = self.get_adain_params(adain_net, w, combined.shape[-1])
		adain1 = self.adain(self.noise(combined), scale, bias)
		synthesized1 = self.deconv1(adain1)
		(scale, bias) = self.get_adain_params(adain_net, w, synthesized1.shape[-1])
		adain2 = self.adain(self.noise(synthesized1), scale, bias)
		activated1 = self.activation(adain2)

		(scale, bias) = self.get_adain_params(adain_net, w, synthesized1.shape[-1])
		adain3 = self.adain(self.noise(activated1), scale, bias)
		synthesized2 = self.deconv2(adain3)
		(scale, bias) = self.get_adain_params(adain_net, w, synthesized2.shape[-1])
		adain4 = self.adain(self.noise(synthesized2), scale, bias)
		activated2 = self.activation(adain4)

		(scale, bias) = self.get_adain_params(adain_net, w, synthesized2.shape[-1])
		adain5 = self.adain(self.noise(activated2), scale, bias)
		synthesized3 = self.deconv3(adain5)
		(scale, bias) = self.get_adain_params(adain_net, w, synthesized3.shape[-1])
		adain6 = self.adain(self.noise(synthesized3), scale, bias)
		activated3 = self.activation(adain6)

		return self.finaldeconv(adain6)

	def get_adain_params(self, model, w, num_channels):
		adain_params = model(w)
		scale = tf.reshape(tf.slice(adain_params, [0, 0, 0], (batch_size, 1, num_channels)), (batch_size, -1))
		bias = tf.reshape(tf.slice(adain_params, [0, 1, 0], (batch_size, 1, num_channels)), (batch_size, -1))
		return scale, bias

class Discriminator_Model(tf.keras.Model):
	def __init__(self):
		super(Discriminator_Model, self).__init__()
		"""
		The model for the discriminator network is defined here. 
		"""
		# cGAN embedding of labels
		self.embedding_size = 50
		self.embedding = Embedding(n_genres, self.embedding_size, name='DiscEmbedding')
		self.embed_dense = Dense(image_side_len**2, activation=leaky_relu)
		self.embed_reshape = Reshape((image_side_len, image_side_len, 1))

		# GAN conv layers
		self.conv1 = Conv2D(num_channels // 8, (5, 5), strides=(2, 2), padding='same', use_bias=False)
		self.conv2 = Conv2D(num_channels // 4, (5, 5), strides=(2, 2), padding='same', use_bias=False)
		self.conv3 = Conv2D(num_channels // 2, (5, 5), strides=(2, 2), padding='same', use_bias=False)
		self.conv4 = Conv2D(num_channels, (5, 5), strides=(2, 2), padding='same', use_bias=False)

		# condense into a decision
		self.decision = Dense(1, activation='sigmoid')

		# self.norm = BatchNormalization()
		self.activate = LeakyReLU(0.2)
		self.flat = Flatten()
		self.merge = Concatenate()

	@tf.function
	def call(self, inputs, genres):
		"""
		Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

		:param inputs: a batch of images, shape=[batch_size, height, width, channels]
		:param genre: a genre string, shape=[1,]

		:return: a batch of values indicating whether the image is real or fake, shape=[batch_size, 1]
		"""
		genres = tf.convert_to_tensor(genres)

		# set up genre embedding
		embed = self.embedding(genres)
		embed = self.embed_reshape(self.embed_dense(embed))

		# merge information
		out = self.merge([embed, inputs])
		# proceed with normal discriminator processes
		out = self.conv1(out)

		# out = self.activate(self.norm(self.conv2(out)))
		out = self.conv2(out)
		(mean, variance) = tf.nn.moments(out, (0, 1, 2))
		out = batch_normalization(out, mean, variance, offset=0, scale=1, variance_epsilon=0.00001)
		
		self.activate(out)

		# out = self.activate(self.norm(self.conv3(out)))
		out = self.conv3(out)
		(mean, variance) = tf.nn.moments(out, (0, 1, 2))
		out = batch_normalization(out, mean, variance, offset=0, scale=1, variance_epsilon=0.00001)
		self.activate(out)

		# out = self.activate(self.norm(self.conv4(out)))
		out = self.conv4(out)
		(mean, variance) = tf.nn.moments(out, (0, 1, 2))
		out = batch_normalization(out, mean, variance, offset=0, scale=1, variance_epsilon=0.00001)
		self.activate(out)

		flat = self.flat(out)
		return self.decision(flat)

class Mapping_Model(tf.keras.Model):
	def __init__(self):
		super(Mapping_Model, self).__init__()
		"""
		Uses a random vector from latent space to learn styles of genres, outputs embeddings.
		Its output is fed into the affine transform layer.
		"""
		self.dense1 = Dense(mapping_dim, use_bias=True, input_shape=(z_dim,), activation=leaky_relu, name='Mapping1')
		self.dense2 = Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping2')
		self.dense3 = Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping3')
		self.dense4 = Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping4')
		self.dense5 = Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping5')
		self.dense6 = Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping6')
		self.dense7 = Dense(mapping_dim, use_bias=True, activation=leaky_relu, name='Mapping7')
		self.dense8 = Dense(z_dim, use_bias=True, name='Mapping8')


	@tf.function
	def call(self, r):
		"""
		"""
		return self.dense8(self.dense7(self.dense6(self.dense5(self.dense4(self.dense3(self.dense2(self.dense1(r))))))))

class ADAin_Model(tf.keras.Model):
	def __init__(self):
		super(ADAin_Model, self).__init__()
		"""
		Uses a random vector from latent space to learn styles of genres, outputs embeddings.
		Its output is fed into the affine transform layer.
		"""
		upspace = num_channels * 4
		self.dense1 = Dense(upspace, use_bias=True, input_shape=(z_dim,), activation=leaky_relu, name='ADAin1')
		self.dense2 = Dense(upspace, use_bias=True, activation=leaky_relu, name='ADAin2')
		self.dense3 = Dense(upspace, use_bias=True, activation=leaky_relu, name='ADAin3')
		self.dense4 = Dense(upspace, use_bias=True, activation=leaky_relu, name='ADAin4')
		self.dense5 = Dense(upspace, use_bias=True, activation=leaky_relu, name='ADAin5')
		self.dense6 = Dense(upspace, use_bias=True, activation=leaky_relu, name='ADAin6')
		self.dense7 = Dense(upspace, use_bias=True, activation=leaky_relu, name='ADAin7')
		self.dense8 = Dense(4 * num_channels, use_bias=True, name='ADAin8')
		self.shape = Reshape((2, 2 * num_channels))
		# model.add(Dense(2 * num_channels, use_bias=True))
		# model.add(Reshape((2, num_channels)))

	@tf.function
	def call(self, w):
		"""
		"""
		return self.shape(self.dense8(self.dense7(self.dense6(self.dense5(self.dense4(self.dense3(self.dense2(self.dense1(w)))))))))

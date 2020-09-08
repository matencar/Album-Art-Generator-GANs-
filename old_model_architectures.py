def make_generator(num_channels):
	"""
	The model for the generator network is defined here. 
	"""
	# # instantiate model
	# model = Sequential()

	# # starts with shape (z_dim * 4 * 4) random input
	# # add noise
	# model.add(GaussianNoise(stddev=1), input_size=(4, 4, z_dim))
	# # do ADAin with vector in W space (from mapping net)
	# model.add(ADAin())
	# # do Conv 3x3
	# model.add(Conv2DTranspose(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# # add noise
	# model.add(GaussianNoise(stddev=1))
	# # do ADAin with vector in W space (from mapping net)
	# model.add(ADAin())

	# # do upsampling
	# # add noise
	# model.add(GaussianNoise(stddev=1))
	# # do ADAin with vector in W space (from mapping net)
	# model.add(ADAin())
	# # do Conv 3x3
	# model.add(Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# # add noise
	# model.add(GaussianNoise(stddev=1))
	# # do ADAin with vector in W space (from mapping net)
	# model.add(ADAin())

	# # continue...

	# # return model
	return Generator_Model(num_channels)


def make_discriminator(num_channels, n_genres, img_shape):
	"""
	The model for the discriminator network is defined here. 
	"""
	# # instantiate model
	# model = Sequential()

	# # conv layers
	# model.add(Conv2D(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False))

	# model.add(Conv2D(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(0.2))

	# model.add(Conv2D(num_channels // 2, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(0.2))

	# model.add(Conv2D(num_channels, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(0.2))

	# # condense into a decision
	# model.add(Flatten())
	# model.add(Dense(1, activation='sigmoid'))

	return Discriminator_Model(num_channels, n_genres, img_shape)


# class Affine_Transform(Model):
# 	def __init__(self, num_channels):
# 		"""
# 		The model that learns scale and bias for ADAin is defined here. 
# 		"""
# 		super(Affine_Transform, self).__init__()
		
# 		upspace = num_channels * 4
# 		self.dense1 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense2 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense3 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense4 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense5 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense6 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense7 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense8 = Dense(2 * num_channels, use_bias=False, activation='softmax'))=
# 		model.add(Reshape((2, num_channels)))
		

# 	@tf.function
# 	def call(self, adain_net, w, noise_scale=1):
# 		"""
# 		Executes the generator model on the random noise vectors.

# 		:param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

# 		:return: prescaled generated images, shape=[batch_size, height, width, channel]
# 		"""
# 		(batch_size, z_dim) = w.shape
# 		(scale, bias) = self.get_adain_params(adain_net, w)
# 		rand_const = tf.random.normal((batch_size, 4, 4, z_dim))
		
# 		adain1 = self.adain(self.noise(rand_const), scale, bias)
# 		synthesized1 = self.deconv1(adain1)
# 		adain2 = self.adain(self.noise(synthesized1), scale, bias)
# 		activated1 = self.activation(adain2)

# 		adain3 = self.adain(self.noise(activated1), scale, bias)
# 		synthesized2 = self.deconv2(adain3)
# 		adain4 = self.adain(self.noise(synthesized2), scale, bias)
# 		activated2 = self.activation(adain4)

# 		adain5 = self.adain(self.noise(activated2), scale, bias)
# 		synthesized3 = self.deconv2(adain5)
# 		adain6 = self.adain(self.noise(synthesized3), scale, bias)

# 		return adain6
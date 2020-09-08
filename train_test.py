import tensorflow as tf
from tensorflow.random import uniform
from tensorflow import GradientTape
import numpy as np

from fid import fid_function
from loss_functions import generator_loss, discriminator_loss

import numpy as np

from imageio import imwrite

from get_args import get_args
args = get_args()
z_dim = args.z_dim
batch_size = args.batch_size
num_channels = args.num_channels
num_genres = args.num_genres
num_gen_updates = args.num_gen_updates
out_dir = args.out_dir

from tensorflow.keras.optimizers import Adam
generator_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)
discriminator_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)
map_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)
adain_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)

# Train the model for one epoch.
def train(
		generator, 
		discriminator, 
		dataset,
		genre_labels, 
		manager, 
		mapping_net,
		noise_net,
		adain_net,
	):
	"""
	Train the model for one epoch. Save a checkpoint every 500 or so batches.

	:param generator: generator model
	:param discriminator: discriminator model
	:param dataset: list of all album covers
	:param manager: the manager that handles saving checkpoints by calling save()

	:return: The average FID score over the epoch
	"""
	sum_fid = 0
	indices = tf.random.shuffle(tf.range(len(genre_labels)))
	num_examples = len(indices)

	# Loop over our data until we run out
	for i in range(num_examples):
		batch = tf.gather(dataset, indices[i : i + batch_size if i + batch_size < num_examples else num_examples])
		labels = tf.gather(genre_labels, indices[i : i + batch_size if i + batch_size < num_examples else num_examples])

		z = uniform((batch_size, z_dim), minval=-1, maxval=1)

		with GradientTape(persistent=True) as tape:
			w = mapping_net(z)

			# generated images
			G_sample = generator(adain_net, w, labels)

			# test discriminator against real images
			logits_real = discriminator(batch, labels)
			# re-use discriminator weights on new inputs
			logits_fake = discriminator(G_sample, labels)

			
			g_loss = generator_loss(logits_fake)
			# g_loss = tf.reduce_sum(p)
			#g_loss = tf.reduce_sum(G_sample)
			d_loss = discriminator_loss(logits_real, logits_fake)

		map_grads = tape.gradient(g_loss, mapping_net.trainable_variables) # success measured by same parameters
		map_optimizer.apply_gradients(zip(map_grads, mapping_net.trainable_variables))

		a_grads = tape.gradient(g_loss, adain_net.trainable_variables) # success measured by same parameters
		adain_optimizer.apply_gradients(zip(a_grads, adain_net.trainable_variables))
			
		# optimize the generator and the discriminator
		gen_gradients = tape.gradient(g_loss, generator.trainable_variables)
		generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

		if (i % num_gen_updates == 0):
			disc_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
			discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

		# Save
		if i % args.save_every == 0:
			manager.save()

		# Calculate inception distance and track the fid in order
		# to return the average
		if i % 500 == 0:
			fid_ = fid_function(batch, G_sample)
			print('**** D_LOSS: %g ****' % d_loss)
			print('**** G_LOSS: %g ****' % g_loss)
			print('**** INCEPTION DISTANCE: %g ****' % fid_)
			sum_fid += fid_
	return sum_fid / (i // 500)


# Test the model by generating some samples.
def test(generator, mapping_net):
	"""
	Test the model.

	:param generator: generator model

	:return: None
	"""
	img = np.array(generator(adain_net, mapping_net(uniform(batch_size, z_dim), minval=-1, maxval=1), np.random.randint(num_genres+1, size=(batch_size,))))

	### Below, we've already provided code to save these generated images to files on disk
	# Rescale the image from (-1, 1) to (0, 255)
	img = ((img / 2) - 0.5) * 255
	# Convert to uint8
	img = img.astype(np.uint8)
	# Save images to disk
	for i in range(0, batch_size):
		img_i = img[i]
		s = out_dir+'/'+str(i)+'.png'
		imwrite(s, img_i)

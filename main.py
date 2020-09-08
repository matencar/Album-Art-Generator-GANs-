# StyleGAN: https://arxiv.org/pdf/1812.04948.pdf
# GANs: https://arxiv.org/pdf/1406.2661.pdf
# Progressive GANs: https://arxiv.org/pdf/1710.10196.pdf
# ADAin: https://arxiv.org/pdf/1703.06868.pdf

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose

from preprocess import load_image_batch
from get_args import get_args
from train_test import train, test
from models import Generator_Model, Discriminator_Model, Mapping_Model, ADAin_Model, make_noise_scale_net

import numpy as np
import os

args = get_args(want_gpu=True)

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		
def main():
	# Load images
	rock = load_image_batch(args.img_dir + '/rock')
	rap = load_image_batch(args.img_dir + '/rap')
	jazz = load_image_batch(args.img_dir + '/jazz')

	# generate labels and make full dataset
	genre_labels = np.zeros(((len(rock) + len(rap) + len(jazz)),))
	dataset = np.zeros(((len(rock) + len(rap) + len(jazz)), rock[0].shape[0], rock[0].shape[1], rock[0].shape[2]))

	# concatenating is super slow, so we do this
	for i in range(len(rock)):
		dataset[i] = rock[i]

	offset = len(rock)
	for i in range(len(rap)):
		genre_labels[i + offset] = 1
		dataset[i + offset] = rap[i]
		
	offset = len(rock) + len(jazz)
	for i in range(len(jazz)):
		genre_labels[i + offset] = 2
		dataset[i + offset] = jazz[i]

	dataset = tf.convert_to_tensor(dataset)
	genre_labels = tf.convert_to_tensor(genre_labels)
	
	# Initialize models
	generator = Generator_Model()
	discriminator = Discriminator_Model()
	mapping_net = Mapping_Model()
	noise_net = make_noise_scale_net()
	adain_net = ADAin_Model()

	# For saving/loading models
	checkpoint_dir = './checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
	manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
	# Ensure the output directory exists
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	if args.restore_checkpoint or args.mode == 'test':
		# restores the latest checkpoint using from the manager
		checkpoint.restore(manager.latest_checkpoint) 

	try:
		# Specify an invalid GPU device
		with tf.device('/device:' + args.device):
			if args.mode == 'train':
				for epoch in range(0, args.num_epochs):
					print('========================== EPOCH %d  ==========================' % epoch)
					avg_fid = train(generator, discriminator, dataset, genre_labels, manager, mapping_net, noise_net, adain_net)
					print("Average FID for Epoch: " + str(avg_fid))
					# Save at the end of the epoch, too
					print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
					manager.save()
			if args.mode == 'test':
				test(generator, args.batch_size, args.z_dim, args.out_dir)
	except RuntimeError as e:
		print(e)

if __name__ == '__main__':
   main()



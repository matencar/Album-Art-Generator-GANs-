import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
cross_entropy = BinaryCrossentropy()

@tf.function
def discriminator_loss(disc_real_output, disc_fake_output):
    """
    Outputs the discriminator loss given the discriminator model output on the real and generated images.

    :param disc_real_output: discriminator output on the real images, shape=[batch_size, 1]
    :param disc_fake_output: discriminator output on the generated images, shape=[batch_size, 1]

    :return: loss, the combined cross entropy loss, scalar
    """
    loss = cross_entropy(tf.zeros_like(disc_fake_output), disc_fake_output)
    loss += cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    return loss


@tf.function
def generator_loss(disc_fake_output):
    """
    Outputs the loss given the discriminator output on the generated images.

    :param disc_fake_output: the discrimator output on the generated images, shape=[batch_size,1]

    :return: loss, the cross entropy loss, scalar
    """
    return cross_entropy(tf.ones_like(disc_fake_output), disc_fake_output)

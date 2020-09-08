import tensorflow as tf
import os
from os import listdir
from os.path import join

def load_image_batch(dir_name, batch_size=128, shuffle_buffer_size=250000, n_threads=2):
    """
    Given a directory, returns a Python list of decoded images in the directory

    :param dir_name: location of images to be decoded

    :return: list of decoded images
    """
    
    def load_and_process_image(filename):
        """
        Given a file path, this function opens and decodes the image stored in the file.

        :param file_path: a batch of images

        :return: an rgb image
        """
        file_path = join(dir_name, filename)
        # Load image
        image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
        # Convert image to normalized float (0, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Rescale data to range (-1, 1)
        image = (image - 0.5) * 2
        return image

    # List file names/file paths
    dataset = listdir(dir_name)

    # Shuffle order
    tf.random.shuffle(dataset)

    # Load and process images
    dataset = list(map(load_and_process_image, dataset))

    return dataset
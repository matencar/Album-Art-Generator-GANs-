import argparse
import tensorflow as tf

def get_args(want_gpu=False):
    gpu_available = False
    if (want_gpu):
        gpu_available = tf.test.is_gpu_available()
        print("GPU Available: ", gpu_available)
    
    parser = argparse.ArgumentParser(description='DCGAN')

    parser.add_argument('--img-dir', type=str, default='./processed_data',
                        help='Data where training images live')

    parser.add_argument('--out-dir', type=str, default='./output',
                        help='Data where sampled output images will be written')

    parser.add_argument('--mode', type=str, default='train',
                        help='Can be "train" or "test"')

    parser.add_argument('--restore-checkpoint', action='store_true',
                        help='Use this flag if you want to resuming training from a previously-saved checkpoint')

    parser.add_argument('--z-dim', type=int, default=512,
                        help='Dimensionality of the latent space')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='Sizes of image batches fed through the network')

    parser.add_argument('--num-data-threads', type=int, default=2,
                        help='Number of threads to use when loading & pre-processing training images')

    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of passes through the training data to make before stopping')

    parser.add_argument('--learn-rate', type=float, default=0.0002,
                        help='Learning rate for Adam optimizer')

    parser.add_argument('--beta1', type=float, default=0.5,
                        help='"beta1" parameter for Adam optimizer')

    parser.add_argument('--num-gen-updates', type=int, default=2,
                        help='Number of generator updates per discriminator update')

    parser.add_argument('--log-every', type=int, default=7,
                        help='Print losses after every [this many] training iterations')

    parser.add_argument('--save-every', type=int, default=500,
                        help='Save the state of the network after every [this many] training iterations')

    parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                        help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

    parser.add_argument('--num-channels', type=int, default=512,
                        help='Output of generator will have [this many] channels')

    parser.add_argument('--clear-processed', type=bool, default=False,
                        help='Clears the ./processed_data directory')

    parser.add_argument('--processed-dir', type=str, default='/default',
                        help='Places newly processed images in ./processed_data/this_directory')

    parser.add_argument('--mapping-dim', type=int, default=512,
                        help='Mapping network dense layers will have [this many] units')

    parser.add_argument('--num-genres', type=int, default=3,
                        help='[This many] genres are analyzed')

    parser.add_argument('--image-side-len', type=int, default=64,
                        help='Length of one side of an image (images must be square)')

    return parser.parse_args()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import tensorflow as tf

from models import Generator
from datasets import read_image, save_image



def main(args):
    translation_net = Generator()
    _ = translation_net(tf.random.normal([1, 256, 256, 3]))
    translation_net.load_weights(args.weights)
    print(f'Weights loaded from {args.weights}.')

    image = read_image(args.image)
    image = tf.cast(image, tf.float32) / 127.5 - 1.0  # from [0, 255] to [-1, 1]
    image = tf.expand_dims(image, axis=0)
    
    translated_image = translation_net(image)[0]
    
    image_name = args.image.split('/')[-1].split('.')[0]
    weights_name = args.weights.split('/')[-1].split('.')[0]
    save_path = f'./output/{image_name}_{weights_name}.jpg'
    save_image(translated_image, save_path)
    print(f'Result saved at {save_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, required=True, help='Path to the image to be translated')
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights file to be loaded')

    args = parser.parse_args()
    main(args)
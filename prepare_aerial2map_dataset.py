import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import tensorflow as tf



image_paths = glob.glob('./maps/*/*.jpg')

for i, path in enumerate(image_paths):
    image = tf.image.decode_png(tf.io.read_file(path), channels=3)  # (600, 1200, 3)
    aerial_image, map_image = image[:, :600, :], image[:, 600:, :]
    
    aerial_image = tf.cast(tf.image.resize(aerial_image, (256, 256), method='bilinear'), 
                           tf.uint8)
    map_image = tf.cast(tf.image.resize(map_image, (256, 256), method='bilinear'), 
                        tf.uint8)
    
    tf.io.write_file(path.replace('maps', 'aerial2map/aerial').replace('val', 'test'), 
                     tf.image.encode_jpeg(aerial_image, quality=100, chroma_downsampling=False))
    tf.io.write_file(path.replace('maps', 'aerial2map/map').replace('val', 'test'), 
                     tf.image.encode_jpeg(map_image, quality=100, chroma_downsampling=False))
    
    if (i + 1) % 100 == 0:
        print(f'[{i+1}/{len(image_paths)}]')
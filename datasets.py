import glob
import random
import tensorflow as tf



def read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image  # uint8


def save_image(image, path):  # image is in [-1, 1]
    image = (image + 1.0) * 127.5
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, tf.uint8)
    tf.io.write_file(path, tf.image.encode_jpeg(image, quality=100, chroma_downsampling=False))


class Dataset:
    def __init__(self, X_path, Y_path):
        self.X_paths_train = sorted(glob.glob(f'{X_path}/train/*.jpg'))
        self.X_paths_test = sorted(glob.glob(f'{X_path}/test/*.jpg'))
        self.X_size_train, self.X_size_test = len(self.X_paths_train), len(self.X_paths_test)
        self.X_paths_train = random.sample(self.X_paths_train, k=self.X_size_train)  # shuffle
        
        self.Y_paths_train = sorted(glob.glob(f'{Y_path}/train/*.jpg'))
        self.Y_paths_test = sorted(glob.glob(f'{Y_path}/test/*.jpg'))
        self.Y_size_train, self.Y_size_test = len(self.Y_paths_train), len(self.Y_paths_test)
        self.Y_paths_train = random.sample(self.Y_paths_train, k=self.Y_size_train)

    def get_train_loader(self):  # returns a generator that allows to iterate over the training images
        X_id, Y_id = 0, 0

        while True:
            if X_id >= self.X_size_train:
                X_id = 0
                self.X_paths_train = random.sample(self.X_paths_train, k=self.X_size_train)  # shuffle

            if Y_id >= self.Y_size_train:
                Y_id = 0
                self.Y_paths_train = random.sample(self.Y_paths_train, k=self.Y_size_train)

            X_image = read_image(self.X_paths_train[X_id])
            X_image = tf.image.random_flip_left_right(X_image)
            X_image = tf.cast(X_image, tf.float32) / 127.5 - 1.0  # from [0, 255] to [-1, 1]
            X_image = tf.expand_dims(X_image, axis=0)  # (1, H, W, 3) float32

            Y_image = read_image(self.Y_paths_train[Y_id])
            Y_image = tf.image.random_flip_left_right(Y_image)
            Y_image = tf.cast(Y_image, tf.float32) / 127.5 - 1.0
            Y_image = tf.expand_dims(Y_image, axis=0)  # (1, H, W, 3) float32
        
            yield X_image, Y_image  # batch consisting of an image from domain X and an image from domain Y
            
            X_id += 1
            Y_id += 1
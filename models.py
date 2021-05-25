import tensorflow as tf
import tensorflow_addons as tfa



class ResBlock(tf.keras.layers.Layer):
    def __init__(self, ch_out):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 3), strides=1, padding='valid', use_bias=False)
        self.in1 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv2 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 3), strides=1, padding='valid', use_bias=False)
        self.in2 = tfa.layers.InstanceNormalization(epsilon=1e-5)

    def call(self, input):
        x = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = tf.nn.relu(self.in1(self.conv1(x)))
        
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = tf.nn.relu(self.in2(self.conv2(x)))
        
        x = tf.keras.layers.add([x, input])
        return x

    def plot(self):
        input = tf.keras.Input(shape=[64, 64, 256])
        model = tf.keras.Model(inputs=input, outputs=self.call(input))
        tf.keras.utils.plot_model(model, to_file='ResBlock.png', show_shapes=True, show_layer_names=False)
        

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=1, padding='valid', use_bias=False)
        self.in1 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)
        self.in2 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)
        self.in3 = tfa.layers.InstanceNormalization(epsilon=1e-5)

        self.resblocks = [
            ResBlock(ch_out=256), ResBlock(ch_out=256), ResBlock(ch_out=256), 
            ResBlock(ch_out=256), ResBlock(ch_out=256), ResBlock(ch_out=256), 
            ResBlock(ch_out=256), ResBlock(ch_out=256), ResBlock(ch_out=256), 
        ]

        self.conv4 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)
        self.in4 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv5 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)
        self.in5 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv6 = tf.keras.layers.Conv2D(3, kernel_size=(7, 7), strides=1, padding='valid', use_bias=False)

    def call(self, input):
        x = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = tf.nn.relu(self.in1(self.conv1(x)))

        x = tf.nn.relu(self.in2(self.conv2(x)))
        
        x = tf.nn.relu(self.in3(self.conv3(x)))

        for block in self.resblocks:
            x = block(x)

        x = tf.nn.relu(self.in4(self.conv4(x)))

        x = tf.nn.relu(self.in5(self.conv5(x)))

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.conv6(x)

        x = tf.tanh(x)  # to [-1, 1]
        return x

    def plot(self):
        input = tf.keras.Input(shape=[256, 256, 3])
        model = tf.keras.Model(inputs=input, outputs=self.call(input))
        tf.keras.utils.plot_model(model, to_file='Generator.png', show_shapes=True, show_layer_names=False)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=2, padding='same', use_bias=False)
        self.in2 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=2, padding='same', use_bias=False)
        self.in3 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv4 = tf.keras.layers.Conv2D(512, kernel_size=(4, 4), strides=1, padding='valid', use_bias=False)
        self.in4 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv5 = tf.keras.layers.Conv2D(1, kernel_size=(4, 4), strides=1, padding='valid')

    def call(self, input):
        x = tf.nn.leaky_relu(self.conv1(input))

        x = tf.nn.leaky_relu(self.in2(self.conv2(x)))
        
        x = tf.nn.leaky_relu(self.in3(self.conv3(x)))

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        x = tf.nn.leaky_relu(self.in4(self.conv4(x)))

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        x = self.conv5(x)

        x = tf.squeeze(x, axis=-1)
        return x

    def plot(self):
        input = tf.keras.Input(shape=[256, 256, 3])
        model = tf.keras.Model(inputs=input, outputs=self.call(input))
        tf.keras.utils.plot_model(model, to_file='Discriminator.png', show_shapes=True, show_layer_names=False)
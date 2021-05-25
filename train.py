import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import tensorflow as tf

from models import Generator, Discriminator
from datasets import Dataset, save_image



class FakeImagePool:  # store a history of generated images
    def __init__(self):
        self.max_size = 50
        self.fake_images = []  # list of (1, 256, 256, 3) float32

    def query(self, fake_image):
        if len(self.fake_images) < self.max_size:  # return the provided generated image, and add it to the pool
            self.fake_images.append(fake_image)
            return fake_image
        
        if random.random() > 0.5:  # return the provided generated image
            return fake_image
        else:  # return a random image from the pool, and replace it by the provided generated image
            random_id = random.randint(0, self.max_size - 1)
            old_fake_image = self.fake_images[random_id]
            self.fake_images[random_id] = fake_image
            return old_fake_image


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, total_steps, step_decay):
        super(CustomSchedule, self).__init__()
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.step_decay = tf.cast(step_decay, tf.float32)

    # the learning rate is initial_lr for step_decay steps, and then it is linearly decayed to 0
    def __call__(self, step):
        return tf.where(tf.less(step, self.step_decay),
                        self.initial_lr,
                        self.initial_lr * (1 - (step - self.step_decay) / (self.total_steps - self.step_decay)))


def main():
    X_PATH = './aerial2map/aerial'
    Y_PATH = './aerial2map/map'
    H, W = 256, 256
    LAMBDA_CYC = 10.0
    NUM_STEPS = 200000
    LOG_EVERY = 100
    SAVE_EVERY = 4000


    weights_path = './weights'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
        print(f'{weights_path} directory created.')

    samples_path = './samples'
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)
        print(f'{samples_path} directory created.')


    dataset = Dataset(X_PATH, Y_PATH)
    train_loader = dataset.get_train_loader()

    G_net = Generator()
    F_net = Generator()
    E_net = Discriminator()
    H_net = Discriminator()

    _ = G_net(tf.random.normal([1, H, W, 3]))
    _ = F_net(tf.random.normal([1, H, W, 3]))
    _ = E_net(tf.random.normal([1, H, W, 3]))
    _ = H_net(tf.random.normal([1, H, W, 3]))
    print('Networks initialized.')

    GF_opt = tf.keras.optimizers.Adam(CustomSchedule(initial_lr=2e-4, total_steps=NUM_STEPS, step_decay=NUM_STEPS//2), beta_1=0.5)
    EH_opt = tf.keras.optimizers.Adam(CustomSchedule(initial_lr=2e-4, total_steps=NUM_STEPS, step_decay=NUM_STEPS//2), beta_1=0.5)


    @tf.function
    def GF_train_step(X, Y):  # (1, H, W, 3), (1, H, W, 3)
        print('Tracing GF_train_step...')

        # Forward pass
        with tf.GradientTape() as tape:
            X2Y = G_net(X)
            Y2X = F_net(Y)
            X2Y2X = F_net(X2Y)
            Y2X2Y = G_net(Y2X)
            p_X2Y = H_net(X2Y)
            p_Y2X = E_net(Y2X)

            G_adv_loss = tf.reduce_mean(tf.math.squared_difference(p_X2Y, 1))
            F_adv_loss = tf.reduce_mean(tf.math.squared_difference(p_Y2X, 1))
            GF_cyc_loss = LAMBDA_CYC * (tf.reduce_mean(tf.math.abs(X - X2Y2X)) + 
                                        tf.reduce_mean(tf.math.abs(Y - Y2X2Y)))
            GF_total_loss = G_adv_loss + F_adv_loss + GF_cyc_loss
            
        # Backward pass
        GF_grads = tape.gradient(GF_total_loss, G_net.trainable_variables + F_net.trainable_variables)
        GF_opt.apply_gradients(zip(GF_grads, G_net.trainable_variables + F_net.trainable_variables))

        return GF_total_loss, GF_cyc_loss, X2Y, Y2X, X2Y2X, Y2X2Y


    @tf.function
    def EH_train_step(X, Y, X2Y, Y2X):  # (1, H, W, 3), (1, H, W, 3), (1, H, W, 3), (1, H, W, 3)
        print('Tracing EH_train_step...')

        # Forward pass
        with tf.GradientTape() as tape:
            p_X = E_net(X)
            p_Y = H_net(Y)
            p_X2Y = H_net(X2Y)
            p_Y2X = E_net(Y2X)

            E_adv_loss = (tf.reduce_mean(tf.math.squared_difference(p_X, 1)) +
                          tf.reduce_mean(tf.math.squared_difference(p_Y2X, 0)))
            H_adv_loss = (tf.reduce_mean(tf.math.squared_difference(p_Y, 1)) +
                          tf.reduce_mean(tf.math.squared_difference(p_X2Y, 0)))
            EH_total_loss = 0.5 * (E_adv_loss + H_adv_loss)

        # Backward pass
        EH_grads = tape.gradient(EH_total_loss, E_net.trainable_variables + H_net.trainable_variables)
        EH_opt.apply_gradients(zip(EH_grads, E_net.trainable_variables + H_net.trainable_variables))

        return EH_total_loss


    X2Y_pool = FakeImagePool()
    Y2X_pool = FakeImagePool()

    for step_id in range(NUM_STEPS):
        X, Y = next(train_loader)
        GF_total_loss, GF_cyc_loss, X2Y, Y2X, X2Y2X, Y2X2Y = GF_train_step(X, Y)
        
        X2Y_new = X2Y_pool.query(X2Y)
        Y2X_new = Y2X_pool.query(Y2X)
        EH_total_loss = EH_train_step(X, Y, X2Y_new, Y2X_new)

        if (step_id + 1) % LOG_EVERY == 0:
            print(f'[Step {step_id+1}/{NUM_STEPS}] GF_total_loss: {GF_total_loss:.4f}. GF_cyc_loss: {GF_cyc_loss:.4f}. EH_total_loss: {EH_total_loss:.4f}.')

        if (step_id + 1) % SAVE_EVERY == 0:
            G_net.save_weights(f'{weights_path}/G{step_id+1}.h5')
            F_net.save_weights(f'{weights_path}/F{step_id+1}.h5')
            E_net.save_weights(f'{weights_path}/E{step_id+1}.h5')
            H_net.save_weights(f'{weights_path}/H{step_id+1}.h5')
            print(f'Weights saved at {weights_path}/[G|F|E|H]{step_id+1}.h5.')
            
            image_to_save = tf.concat([tf.concat([X, X2Y, X2Y2X], axis=2), 
                                       tf.concat([Y, Y2X, Y2X2Y], axis=2)], axis=1)[0]
            save_image(image_to_save, f'{samples_path}/{step_id+1}.jpg')
            print(f'Sample images saved at {samples_path}/{step_id+1}.jpg.')
            

if __name__ == '__main__':
    main()
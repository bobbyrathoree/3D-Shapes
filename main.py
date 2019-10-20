import os
import time

from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from callback import log_this
from discriminator import build_discriminator
from generator import build_generator
from loading_and_visualizing import get_three_d_images, save_voxels
import numpy as np


def main():
    object_name = "chair"
    data_dir = "data/3DShapeNets/volumetric_data/" \
               "{}/30/train/*.mat".format(object_name)
    gen_learning_rate = 0.0025
    dis_learning_rate = 10e-5
    beta = 0.5
    batch_size = 1
    z_size = 200
    epochs = 10
    MODE = "train"

    """
    Create models
    """
    gen_optimizer = Adam(lr=gen_learning_rate, beta_1=beta)
    dis_optimizer = Adam(lr=dis_learning_rate, beta_1=beta)

    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    discriminator.trainable = False

    input_layer = Input(shape=(1, 1, 1, z_size))
    generated_volumes = generator(input_layer)
    validity = discriminator(generated_volumes)
    adversarial_model = Model(inputs=[input_layer], outputs=[validity])
    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    print("Loading data...")
    volumes = get_three_d_images(directory=data_dir)
    volumes = volumes[..., np.newaxis].astype(np.float)
    print("Data loaded...")

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    labels_real = np.reshape(np.ones((batch_size,)), (-1, 1, 1, 1, 1))
    labels_fake = np.reshape(np.zeros((batch_size,)), (-1, 1, 1, 1, 1))

    if MODE == 'train':
        for epoch in range(epochs):
            print("Epoch:", epoch)

            gen_losses = []
            dis_losses = []

            number_of_batches = int(volumes.shape[0] / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)

                z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                volumes_batch = volumes[index * batch_size:(index + 1) * batch_size, :, :, :]

                # Next, generate volumes using the generate network
                gen_volumes = generator.predict_on_batch(z_sample)

                """
                Train the discriminator network
                """
                discriminator.trainable = True
                if index % 2 == 0:
                    loss_real = discriminator.train_on_batch(volumes_batch, labels_real)
                    loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)

                    d_loss = 0.5 * np.add(loss_real, loss_fake)
                    print("d_loss:{}".format(d_loss))

                else:
                    d_loss = 0.0

                discriminator.trainable = False
                """
                Train the generator network
                """
                z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                g_loss = adversarial_model.train_on_batch(z, labels_real)
                print("g_loss:{}".format(g_loss))

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

                # Every 10th mini-batch, generate volumes and save them
                if index % 10 == 0:
                    z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                    generated_volumes = generator.predict(z_sample2, verbose=3)
                    for i, generated_volume in enumerate(generated_volumes[:5]):
                        voxels = np.squeeze(generated_volume)
                        voxels[voxels < 0.5] = 0.
                        voxels[voxels >= 0.5] = 1.
                        save_voxels(voxels, "results/img_{}_{}_{}".format(epoch, index, i))

            # Write losses to Tensorboard
            log_this(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
            log_this(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

        """
        Save models
        """
        generator.save_weights(os.path.join("models", "generator_weights.h5"))
        discriminator.save_weights(os.path.join("models", "discriminator_weights.h5"))

    if MODE == 'predict':
        # Create models
        generator = build_generator()
        discriminator = build_discriminator()

        # Load model weights
        generator.load_weights(os.path.join("models", "generator_weights.h5"), True)
        discriminator.load_weights(os.path.join("models", "discriminator_weights.h5"), True)

        # Generate 3D models
        z_sample = np.random.normal(0, 1, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
        generated_volumes = generator.predict(z_sample, verbose=3)

        for i, generated_volume in enumerate(generated_volumes[:2]):
            voxels = np.squeeze(generated_volume)
            voxels[voxels < 0.5] = 0.
            voxels[voxels >= 0.5] = 1.
            save_voxels(voxels, "results/gen_{}".format(i))


if __name__ == "__main__":
    main()

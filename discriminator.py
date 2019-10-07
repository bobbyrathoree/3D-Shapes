from keras import Input, Model
from keras.layers import Conv3D, BatchNormalization, Activation, LeakyReLU


def build_discriminator():
    input_shape = (64, 64, 64, 1)
    kernel_sizes = [64, 128, 256, 512, 1]
    filters = [4, 4, 4, 4, 4]
    strides = [2, 2, 2, 2, 1]
    paddings = ["same", "same", "same", "same", "valid"]
    alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    activations = ["leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu", "sigmoid"]
    convolutional_blocks = 5

    input_layer = Input(shape=input_shape)

    # First 3D Convolutional block
    a = Conv3D(
        filters=filters[0],
        kernel_size=kernel_sizes[0],
        strides=strides[0],
        padding=paddings[0],
    )(input_layer)
    a = BatchNormalization()(a, training=True)
    a = LeakyReLU(alpha=alphas[0])(a)

    # Next 4 3D Convolutional blocks
    for x in range(convolutional_blocks - 1):
        a = Conv3D(
            filters=filters[x + 1],
            kernel_size=kernel_sizes[x + 1],
            strides=strides[x + 1],
            padding=paddings[x + 1],
        )(a)
        a = BatchNormalization()(a, training=True)
        a = (
            LeakyReLU(alpha=alphas[x + 1])
            if activations[x + 1] == "leaky_relu"
            else Activation(activation="sigmoid")
        )(a)

    discriminator_model = Model(inputs=[input_layer], outputs=[a])
    return discriminator_model

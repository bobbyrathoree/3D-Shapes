from keras import Model
from keras.layers import Input, Deconv3D, BatchNormalization, Activation


def build_generator():
    z_size = 200
    filters = [512, 256, 128, 64, 1]
    kernel_sizes = [4, 4, 4, 4, 4]
    strides = [1, 2, 2, 2, 2]
    input_shape = (1, 1, 1, z_size)
    activations = ["relu", "relu", "relu", "relu", "sigmoid"]
    convolutional_blocks = 5

    input_layer = Input(shape=input_shape)

    # First 3D deconvolution block
    a = Deconv3D(filters=filters[0],
                 kernel_size=kernel_sizes[0],
                 strides=strides[0])(input_layer)
    a = BatchNormalization()(a, training=True)
    a = Activation(activation=activations[0])(a)

    # Next 4 will also be 3D deconvolution blocks
    for x in range(convolutional_blocks - 1):
        a = Deconv3D(
            filters=filters[x + 1],
            kernel_size=kernel_sizes[x + 1],
            strides=strides[x + 1],
        )(a)
        a = BatchNormalization()(a, training=True)
        a = Activation(activation=activations[x + 1])(a)

    gen_model = Model(inputs=[input_layer], outputs=[a])

    return gen_model

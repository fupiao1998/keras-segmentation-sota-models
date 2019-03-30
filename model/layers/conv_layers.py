from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Conv2DTranspose
from keras.layers import add
from keras.layers import concatenate
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D


def Conv_bn_relu(num_filters,
                 kernel_size,
                 stage,
                 activation='relu',
                 batchnorm=True,
                 strides=(1, 1),
                 name='conv_block',
                 padding='same'):

    def layer(input_tensor):
        block_name = name + stage
        x = Conv2D(num_filters, kernel_size, use_bias=not (batchnorm),
                   padding=padding, kernel_initializer='he_normal', strides=strides,
                   name=block_name + '_conv')(input_tensor)
        if batchnorm:
            x = BatchNormalization(name=block_name + '_bn', )(x)
        x = Activation(activation, name=block_name + '_' + activation)(x)

        return x
    return layer


def resnet_block(num_filters, kernel_size, stage):

    def layer(input_tensor):
        res = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(input_tensor)
        x = Conv_bn_relu(num_filters//2, kernel_size, stage=stage+'a', activation='relu', batchnorm=True)(input_tensor)
        x = Conv2D(num_filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
        out = add([x, res])

        return out
    return layer


def dense_block(num_filters, kernel_size, stage):

    def layer(input_tensor):
        x1 = Conv_bn_relu(num_filters, kernel_size, stage=stage+'a', activation='relu', batchnorm=True)(input_tensor)
        x1 = concatenate([input_tensor, x1], axis=3)
        x2 = Conv_bn_relu(num_filters, kernel_size, stage=stage + 'b', activation='relu', batchnorm=True)(x1)
        x2 = concatenate([input_tensor, x2], axis=3)

        return x2
    return layer


def xception_block(num_filters, kernel_size, skip):
    def layer(input_tensor):
        x = SeparableConv2D(num_filters, kernel_size, padding='same', use_bias=False)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(num_filters, kernel_size, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = add([x, skip])

        return x
    return layer


def xception_loop(num_filters, kernel_size):
    def layer(input_tensor):
        x = Activation('relu')(input_tensor)
        x = SeparableConv2D(num_filters, kernel_size, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(num_filters, kernel_size, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(num_filters, kernel_size, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        return x
    return layer


def Upsample2D_block(num_filters,
                     stage, kernel_size=(3,3),
                     upsample_rate=(2,2),
                     batchnorm=False, skip=None):
    def layer(input_tensor):
        x = UpSampling2D(size=upsample_rate, name='up_name')(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv_bn_relu(num_filters, kernel_size, stage=stage, batchnorm=batchnorm)(x)

        x = Conv_bn_relu(num_filters, kernel_size, stage=stage+1, batchnorm=batchnorm,)(x)

        return x
    return layer


def Transpose2D_block(num_filters,
                     stage, kernel_size=(3,3),
                     upsample_rate=(2,2),
                     transpose_kernel_size=(4,4),
                     batchnorm=False, skip=None):
    def layer(input_tensor):
        x = Conv2DTranspose(num_filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name='transpose'+str(stage), use_bias=not (batchnorm))(input_tensor)
        if batchnorm:
            x = BatchNormalization(name='bn' + str(stage))(x)
        x = Activation('relu')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv_bn_relu(num_filters, kernel_size, stage=stage, batchnorm=batchnorm)(x)

        return x
    return layer
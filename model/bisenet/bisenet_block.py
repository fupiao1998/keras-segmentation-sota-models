from model.layers.conv_layers import Conv_bn_relu
from model.layers.conv_layers import xception_block
from model.layers.conv_layers import xception_loop
from keras.layers import multiply
from keras.layers import GlobalAvgPool2D
from keras.layers import AveragePooling2D
from keras.layers import Add, Input, add
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.layers import UpSampling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model


def sptial_path(num_filters, kernel_size, strides, stage, padding='valid'):
    def layer(input_tensor):
        x = Conv_bn_relu(num_filters, kernel_size, strides=strides, stage=stage+'a', padding=padding)(input_tensor)
        x = Conv_bn_relu(num_filters, kernel_size, strides=strides, stage=stage + 'b', padding=padding)(x)
        x = Conv_bn_relu(num_filters, kernel_size, strides=strides, stage=stage + 'c', padding=padding)(x)
        return x
    return layer


def ARM_module(num_filters, stage, kernel_size=1):
    def layer(input_tensor):
        # x = GlobalAvgPool2D(dim_ordering='default')(input_tensor)
        x = AveragePooling2D(pool_size=(1, 1), padding='same')(input_tensor)
        x = Conv_bn_relu(num_filters, kernel_size, stage=stage + '_ARM_', activation='sigmoid')(x)
        out = multiply([input_tensor, x])
        return out
    return layer


def FFM_module(num_filters, stage, kernel_size):
    def layer(SP_input, ARM_input):
        x = Concatenate(axis=-1)([SP_input, ARM_input])
        x = Conv_bn_relu(num_filters, kernel_size, stage=stage + 'a')(x)
        x1 = AveragePooling2D(pool_size=(1, 1), padding='same')(x)
        x1 = Conv_bn_relu(num_filters, kernel_size=(1, 1), batchnorm=False, stage=stage + 'relu')(x1)
        x1 = Conv_bn_relu(num_filters, kernel_size=(1, 1), batchnorm=False, stage=stage + 'sigmoid', activation='sigmoid')(x1)
        x2 = multiply([x1, x])
        x3 = Add()([x, x2])

        return x3
    return layer


def xception():
    def layer(input_tensor):
        x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(input_tensor)
        x = Conv2D(64, (3, 3))(x)
        # block1
        res1 = Conv2D(128, (1, 1), strides=(2, 2))(x)
        xce1 = xception_block(128, (3, 3), res1)(x)
        # block2
        xce1 = Activation('relu')(xce1)
        res2 = Conv2D(256, (1, 1), strides=(2, 2))(xce1)
        xce2 = xception_block(256, (3, 3), res2)(xce1)
        # block3
        xce2 = Activation('relu')(xce2)
        res3 = Conv2D(728, (1, 1), strides=(2, 2))(xce2)
        xce = xception_block(728, (3, 3), res3)(xce2)
        # loop 8 times
        for i in range(8):
            res = xce
            xce = xception_loop(728, (3, 3))(xce)
            xce = add([res, xce])
        # final block
        xce_fin = Activation('relu')(xce)
        res_fin = Conv2D(1024, (1, 1), strides=(2, 2))(xce_fin)
        xce_fin = xception_block(1024, (3, 3), res_fin)(xce_fin)
        return xce, xce_fin
    return layer


def context_path():
    def layer(input_tensor):
        down16, down32 = xception()(input_tensor) # down 16 728, down 32 1024
        block1 = ARM_module(728, stage='a', kernel_size=1)(down16)
        block2 = ARM_module(1024, stage='b', kernel_size=1)(down32)
        global_channels = GlobalAveragePooling2D()(block2)
        block2_scaled = multiply([global_channels, block2])
        block1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(block1)
        block2_scaled = UpSampling2D(size=(4, 4), interpolation='bilinear')(block2_scaled)
        cnc = Concatenate(axis=-1)([block1, block2_scaled])
        return cnc
    return layer
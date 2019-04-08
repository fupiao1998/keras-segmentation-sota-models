from model.layers.conv_layers import Bn_relu_conv
from model.layers.conv_layers import resnet_block
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers import add
from keras.layers import Conv2D


def gcn_block(num_filters, kernel_size=9):
    def layer(inpurt_tensor):
        conv_l1 = Conv2D(num_filters, kernel_size=(kernel_size, 1), padding='same')(inpurt_tensor)
        conv_l2 = Conv2D(num_filters, kernel_size=(1, kernel_size), padding='same')(conv_l1)
        conv_r1 = Conv2D(num_filters, kernel_size=(1, kernel_size), padding='same')(inpurt_tensor)
        conv_r2 = Conv2D(num_filters, kernel_size=(kernel_size, 1), padding='same')(conv_r1)
        x = add([conv_l2, conv_r2])
        return x
    return layer


def br_block(num_filters=21, stage='br_1_'):
    def layer(input_tensor):
        res = input_tensor
        x = Bn_relu_conv(num_filters=num_filters, kernel_size=3, stage=stage+'a')(input_tensor)
        x = Bn_relu_conv(num_filters=num_filters, kernel_size=3, stage=stage+'b')(x)
        out = add([res, x])
        return out
    return layer


def GCN_BR(num_filters=21, kernel_size=9, stage='1'):
    def layer(input_tensor):
        x = gcn_block(num_filters, kernel_size)(input_tensor)
        x = br_block(num_filters, stage)(x)
        return x
    return layer


def BR_deconv(num_filters=21, upsample_rate=(2,2), stage='br_deconv_'):
    def layer(input_tensor):
        x = br_block(num_filters, stage)(input_tensor)
        x = Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=upsample_rate, padding='same')(x)
        return x
    return layer


def res_pool(num_filters, kernel_size, stage):
    def layer(input_tensor):
        x = resnet_block(num_filters, kernel_size, stage)(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x
    return layer
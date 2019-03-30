from model.unet.unet_block import conv_pool_block
from model.unet.unet_block import Upsample2D_block
from model.unet.unet_block import Transpose2D_block
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import *


def encoder(num_filters):
    def layer(input_tensor):
        block1 = conv_pool_block(num_filters[0], 3, batchnorm=True, stage=1)(input_tensor)
        # block2 128
        block2 = conv_pool_block(num_filters[1], 3, batchnorm=True, stage=2)(block1)
        # block3 256
        block3 = conv_pool_block(num_filters[2], 3, batchnorm=True, stage=3)(block2)
        # block4 512
        block4 = conv_pool_block(num_filters[3], 3, batchnorm=True, stage=4)(block3)
        # block5 1024
        block5 = conv_pool_block(num_filters[4], 3, batchnorm=True, stage=5)(block4)
        return block5, block4, block3, block2, block1
    return layer


def decoder(num_filters, block4, block3, block2, block1):
    def layer(input_tensor):
        # up1 512
        up1 = Upsample2D_block(num_filters[3], stage=1, batchnorm=True, skip=block4)(input_tensor)
        # up2 256
        up2 = Upsample2D_block(num_filters[2], stage=2, batchnorm=True, skip=block3)(up1)
        # up2 128
        up3 = Upsample2D_block(num_filters[1], stage=3, batchnorm=True, skip=block2)(up2)
        # up3 64
        up4 = Upsample2D_block(num_filters[0], stage=4, batchnorm=True, skip=block1)(up3)
        return up4
    return layer


def unet_normal(img_rows, img_cols, num_filters):
    inputs = Input((img_rows, img_cols, 3))
    block5, block4, block3, block2, block1 = encoder(num_filters)(inputs)
    decoder_out = decoder(num_filters, block4, block3, block2, block1)(block5)
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(decoder_out)
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    output = Conv2D(1, 1, activation='sigmoid')(conv)
    unet = Model(input=inputs, output=output)
    unet.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return unet
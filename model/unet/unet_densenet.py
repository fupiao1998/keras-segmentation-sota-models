from model.unet.unet_block import dense_pool_block
from model.unet.unet_block import Upsample2D_block
from model.unet.unet_block import Transpose2D_block
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import *


def unet_densenet(img_rows, img_cols, num_filters):
    inputs = Input((img_rows, img_cols, 3))
    # block1 64
    block1 = dense_pool_block(num_filters[0], 3, stage=1)(inputs)
    # block2 128
    block2 = dense_pool_block(num_filters[1], 3, stage=2)(block1)
    # block3 256
    block3 = dense_pool_block(num_filters[2], 3, stage=3)(block2)
    # block4 512
    block4 = dense_pool_block(num_filters[3], 3, stage=4)(block3)
    # block5 1024
    block5 = dense_pool_block(num_filters[4], 3, stage=5)(block4)
    # up1 512
    up1 = Upsample2D_block(num_filters[3], stage=1, batchnorm=True, skip=block4)(block5)
    # up2 256
    up2 = Upsample2D_block(num_filters[2], stage=2, batchnorm=True, skip=block3)(up1)
    # up2 128
    up3 = Upsample2D_block(num_filters[1], stage=3, batchnorm=True, skip=block2)(up2)
    # up3 64
    up4 = Upsample2D_block(num_filters[0], stage=4, batchnorm=True, skip=block1)(up3)
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up4)
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    output = Conv2D(1, 1, activation='sigmoid')(conv)
    unet_densenet = Model(input=inputs, output=output)
    unet_densenet.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return unet_densenet


model=unet_densenet(256, 256, num_filters=[64, 128, 256, 512, 1024])
model.summary()
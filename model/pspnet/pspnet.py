from model.layers.conv_layers import ResNet50
from model.layers.conv_layers import Conv_bn_relu
from model.pspnet.pspnet_block import pymraid
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.optimizers import *


def pspnet(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 3))
    resnet_out = ResNet50()(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(resnet_out)
    x = pymraid()(x)
    x = Conv_bn_relu(512, 3, '_py6_')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = concatenate([x, resnet_out])
    output = Conv2D(filters=2, kernel_size=1, strides=1, padding='same', name='sum_conv_2',
                     activation='softmax')(x)
    pspnet = Model(input=inputs, output=output)
    pspnet.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    pspnet.summary()
    return


pspnet(480, 480)
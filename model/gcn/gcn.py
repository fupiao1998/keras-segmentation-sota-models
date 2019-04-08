from model.layers.conv_layers import resnet_block
from model.gcn.gcn_block import gcn_block
from model.gcn.gcn_block import br_block
from model.gcn.gcn_block import GCN_BR
from model.gcn.gcn_block import BR_deconv
from model.gcn.gcn_block import res_pool
from keras.layers import Input
from keras.layers import add
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.models import Model
from keras.optimizers import *


def gcn_net(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 3))
    conv_1 = Conv2D(filters=64, kernel_size=(7, 7), padding='same')(inputs)
    res_2 = res_pool(num_filters=256, kernel_size=3, stage='res1')(conv_1)
    res_3 = res_pool(num_filters=512, kernel_size=3, stage='res2')(res_2)
    res_4 = res_pool(num_filters=1024, kernel_size=3, stage='res3')(res_3)
    res_5 = res_pool(num_filters=2048, kernel_size=3, stage='res4')(res_4)
    gcn_br_1 = GCN_BR(num_filters=21, kernel_size=9, stage='gcn_br_1')(res_2)
    gcn_br_2 = GCN_BR(num_filters=21, kernel_size=9, stage='gcn_br_2')(res_3)
    gcn_br_3 = GCN_BR(num_filters=21, kernel_size=9, stage='gcn_br_3')(res_4)
    gcn_br_4 = GCN_BR(num_filters=21, kernel_size=9, stage='gcn_br_4')(res_5)
    deconv_1 = Conv2DTranspose(21, kernel_size=(2, 2), strides=(2, 2), padding='same')(gcn_br_4)
    add_1 = add([deconv_1, gcn_br_3])
    deconv_2 = BR_deconv(stage='dr_de_1')(add_1)
    add_2 = add([deconv_2, gcn_br_2])
    deconv_3 = BR_deconv(stage='dr_de_2')(add_2)
    add_3 = add([deconv_3, gcn_br_1])
    deconv_4 = BR_deconv(stage='dr_de_3')(add_3)
    deconv_5 = BR_deconv(stage='dr_de_4')(deconv_4)
    output = br_block()(deconv_5)
    output = Conv2D(1, 1, activation='sigmoid')(output)
    gcn = Model(input=inputs, output=output)
    gcn.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    gcn.summary()
    return gcn_net


gcn_net(512, 512)

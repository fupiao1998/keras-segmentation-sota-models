from model.bisenet.bisenet_block import *
from keras.layers import UpSampling2D
from keras.optimizers import *


def bisenet(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 3))
    sp_out = sptial_path(num_filters=128, kernel_size=(2, 2), strides=(2, 2), stage='_sp_layer_')(inputs)
    con_out = context_path()(inputs)
    FFM_out = FFM_module(256, stage='_ffm_', kernel_size=(3, 3))(sp_out, con_out)
    out = UpSampling2D(size=8, name='up_name')(FFM_out)
    bisenet = Model(inputs, out)
    bisenet.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    bisenet.summary()
    return bisenet
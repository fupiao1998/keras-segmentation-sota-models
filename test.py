import numpy as np
from keras.models import load_model
from keras.preprocessing.image import array_to_img
from train import *
import argparse

'''
-name unet_crack.h5
-npath D:\\all-Pythoncodes\\python_for_fun\\Keras_U-net\\npy_data
-rpath D:\\all-Pythoncodes\\python_for_fun\\Keras_U-net\\results
'''
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-name","--model_name", required=True,
                    help="output of model name")
    ap.add_argument("-npath", "--npy_path", required=True,
                    help="path to .npy files")
    ap.add_argument("-rpath", "--result_path", required=True,
                    help="path to result of predictions")
    args = vars(ap.parse_args())
    return args

def load_test_data(npy_path):
    # Load .npy files for test
    imgs_test = np.load(npy_path+"/imgs_test.npy")
    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255
    return imgs_test


def test(model_name, npy_path, result_path):
    model = load_model(model_name)
    imgs_test = load_test_data(npy_path)
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
    np.save(result_path + '\\imgs_mask_test.npy', imgs_mask_test)
    imgs = np.load(result_path + '\\imgs_mask_test.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save(result_path + "\\%d.jpg" % (i+100))


if __name__ == "__main__":
    args = args_parse()
    npy_path = args["npy_path"]
    result_path = args["result_path"]
    name = args["model_name"]
    test(name, npy_path, result_path)
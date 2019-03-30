from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import argparse


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtrain", "--data_path", required=True,
                    help="path to input image")
    ap.add_argument("-dlabel", "--label_path", required=True,
                    help="path to input label")
    ap.add_argument("-dtest", "--test_path", required=True,
                    help="path to test image")
    ap.add_argument("-npath", "--npy_path", required=True,
                    help="path to .npy files")
    ap.add_argument("-itype", "--img_type", required=True,
                    help="path to output model")
    ap.add_argument("-r", "--rows", required=True, type=int,
                    help="shape of rows of input image")
    ap.add_argument("-c", "--cols", required=True, type=int,
                    help="shape of cols of input image")
    args = vars(ap.parse_args())
    return args
'''
-dtrain D:\\all-PythonCodes\\python_for_fun\\Keras_U-net\\data\\train_traffic\\image
-dlabel D:\\all-PythonCodes\\python_for_fun\\Keras_U-net\\data\\train_traffic\\label
-dtest D:\\all-PythonCodes\\python_for_fun\\Keras_U-net\\data\\train_traffic\\test
-npath D:\\all-Pythoncodes\\python_for_fun\\Keras_U-net\\npy_data
-itype png
-r 360
-c 480
'''
'''
-dtrain D:\\all-PythonCodes\\RCFs\\RCF-keras\\building_data\\111\\train
-dlabel D:\\all-PythonCodes\\RCFs\\RCF-keras\\building_data\\111\\label
-dtest D:\\all-PythonCodes\\RCFs\\RCF-keras\\building_data\\111\\test
-npath D:\\all-PythonCodes\\RCFs\\RCF-keras\\building_data\\111
-itype png
-r 256
-c 256
'''


def create_train_data(data_path, img_type, rows, cols, label_path, npy_path):
    # Generate npy files for training sets and labels
    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    imgs = glob.glob(data_path + "//*." + img_type)
    print(len(imgs))
    imgdatas = np.ndarray((len(imgs), rows, cols, 3), dtype=np.uint8)
    imglabels = np.ndarray((len(imgs), rows, cols, 1), dtype=np.uint8)
    for imgname in imgs:
        midname = imgname[imgname.rindex("\\") + 1:]
        img = load_img(data_path + "\\" + midname)
        img = img_to_array(img)
        label = load_img(label_path + "\\" + midname, grayscale=True)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, len(imgs)))
        i += 1
    np.save(npy_path + '\\imgs_train.npy', imgdatas)
    np.save(npy_path + '\\imgs_mask_train.npy', imglabels)
    print('Saving to .npy files done.')


def create_test_data(test_path, img_type, rows, cols, npy_path):
    # Generate npy files for test sets
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    imgs = glob.glob(test_path+"/*."+img_type)
    print(len(imgs))
    imgdatas = np.ndarray((len(imgs), rows, cols,1), dtype=np.uint8)
    for imgname in imgs:
        midname = imgname[imgname.rindex("\\")+1:]
        img = load_img(test_path + "\\" + midname, grayscale = True)
        img = img_to_array(img)
        imgdatas[i] = img
        i += 1
    np.save(npy_path + '\\imgs_test.npy', imgdatas)
    print('Saving to imgs_test.npy files done.')



if __name__ == "__main__":
    args = args_parse()
    data_path = args["data_path"]
    label_path = args["label_path"]
    test_path = args["test_path"]
    npy_path = args["npy_path"]
    img_type = args["img_type"]
    rows = args["rows"]
    cols = args["cols"]
    create_train_data(data_path, img_type, rows, cols, label_path, npy_path)
    create_test_data(test_path, img_type, rows, cols, npy_path)
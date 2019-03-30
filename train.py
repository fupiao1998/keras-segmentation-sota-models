from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from Unet import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
from model.unet import *
'''
-npath D:\\all-Pythoncodes\\python_for_fun\\Keras_U-net\\npy_data
-r 512
-c 512
-name unet_cell_densenet.h5
-ptrain 0
'''


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-npath", "--npy_path", required=True,
                    help="path to .npy files")
    ap.add_argument("-r", "--rows", required=True, type=int,
                    help="shape of rows of input image")
    ap.add_argument("-c", "--cols", required=True, type=int,
                    help="shape of cols of input image")
    ap.add_argument("-name","--model_name", required=True,
                    help="output of model name")
    ap.add_argument("-ptrain", "--pretrain", required=True, type=int,
                    help="if using ptrain model")
    args = vars(ap.parse_args())
    return args


def load_train_data(npy_path):
    # Load .npy files for training
    imgs_train = np.load(npy_path+"\\imgs_train.npy")
    imgs_mask_train = np.load(npy_path+"\\imgs_mask_train.npy")
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_train /= 255
    imgs_mask_train /= 255
    imgs_mask_train[imgs_mask_train > 0.5] = 1
    imgs_mask_train[imgs_mask_train <= 0.5] = 0
    return imgs_train,imgs_mask_train


def train(npy_path, img_rows, img_cols, model_name, ptrain):
    imgs_train, imgs_mask_train = load_train_data(npy_path)
    if ptrain:
        model = load_model(model_name)
    else:
        model = unet_normal(img_rows, img_cols)
    Batch_size=1
    Epochs=50
    model_checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
    model.summary()
    H = model.fit(imgs_train, imgs_mask_train, batch_size=Batch_size, epochs=Epochs, verbose=1, validation_split=0.2,
                        shuffle=True, callbacks=[model_checkpoint])
    #loss_history = H.history["loss"]
    #acc_history = H.history["acc"]
    #numpy_loss_history = np.array(loss_history)
    #np.savetxt("loss_history.txt", numpy_loss_history, fmt="%.18f", delimiter="\n")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, Epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, Epochs), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss on cell seg by denseunet")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig("plot_cell_loss_denseunet.png")
    plt.figure()
    plt.plot(np.arange(0, Epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, Epochs), H.history["val_acc"], label="val_acc")
    plt.title("Training Accuracy on cell seg by denseunet")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot_cell_acc_denseunet.png")


if __name__ == "__main__":
    args = args_parse()
    npy_path = args["npy_path"]
    rows = args["rows"]
    cols = args["cols"]
    name = args["model_name"]
    ptrain = args["pretrain"]
    train(npy_path, rows, cols, name, ptrain)
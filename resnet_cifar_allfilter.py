# Training procedure for CIFAR-10 using ResNet 101.

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
from constants import *
from resnet_class.resnet101 import ResNet_101
from pathlib import Path

if __name__ == '__main__':
    # load data

    train_path = os.path.join("DataSets", "cifar5m_part0")
    valid_path = os.path.join("DataSets", "cifar5m_part1")
    '''
    train_path = os.path.join("..", "UT_OoD", "DataSets", "train_set", "cifar5m_part0")
    valid_path = os.path.join("..", "UT_OoD", "DataSets", "train_set", "cifar5m_part1")
    '''
    for seed in SEEDS:
        filepath = os.path.join("model_weights", "resnet_101_allfilter_10_" + str(seed))
        Path(filepath).mkdir(parents=True, exist_ok=True)
        model_history = os.path.join(filepath, "model_history.p")

        # data_path = os.path.join("cifar5m", "train_set", "cifar5m_part0")

        train_file = "sample_3k_X.npy"
        train_label = "sample_3k_Y.npy"

        x_train = np.load(os.path.join(train_path, train_file))
        y_train = np.load(os.path.join(train_path, train_label))
        x_val = np.load(os.path.join(valid_path, train_file))
        y_val = np.load(os.path.join(valid_path, train_label))

        x_filter = []
        y_filter = []
        fil_valid_data = []
        fil_valid_label = []
        filter_list = ["blur", "detail", "edge_enhance", "smooth", "sharp"]

        for filter in filter_list:
            filter_file = filter + "_1k_X.npy"
            filter_label = filter + "_1k_Y.npy"
            fil_data = np.load(os.path.join(valid_path, filter_file))
            fil_label = np.load(os.path.join(valid_path, filter_label))

            fil_data, fil_vdata, fil_label, fil_vlabel = train_test_split(fil_data, fil_label, test_size=500,
                                                                          random_state=seed)
            fil_valid_data.append(fil_vdata)
            fil_valid_label.append(fil_vlabel)

            # print("blur_label shape: ", blur_label.shape)
            # print(blur.shape)

            for i in range(10):
                temp_x = fil_data[fil_label == i]
                np.random.seed(seed)
                random_indices = np.random.choice(temp_x.shape[0], size=FILTER_SAMPLES, replace=False)
                temp_x = temp_x[random_indices, :]
                # print("temp_x shape: ", temp_x.shape)
                x_filter.append(temp_x)
                temp_y = np.full(temp_x.shape[0], i, dtype=int)
                # ("temp_y shape: ", temp_y.shape)
                y_filter.append(temp_y)
                # print("Y_blur shape: ", np.array(y_blur).shape)

        x_filter = np.concatenate(x_filter)
        y_filter = np.concatenate(y_filter)
        fil_valid_data = np.concatenate(fil_valid_data)
        fil_valid_label = np.concatenate(fil_valid_label)

        y_filter.reshape(y_filter.shape[0], 1)
        # print(x_filter.shape)
        # print("Y_blur shape after concate: ", np.array(y_blur).shape)

        print("Data shape before split: ", x_train.shape)

        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=1000,
                                                            random_state=seed)

        print("Data shape after split: ", x_train.shape)

        _, x_val, _, y_val = train_test_split(x_val, y_val, test_size=500,
                                              random_state=seed)  # random_state = seed

        x_train = np.concatenate([x_train, x_filter])
        y_train = np.concatenate([y_train, y_filter])
        x_val = np.concatenate([x_val, fil_valid_data])
        y_val = np.concatenate([y_val, fil_valid_label])

        y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
        y_val = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

        print("x_train shape: ", x_train.shape)
        print("y_train shape: ", y_train.shape)

        img_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
        resnet = ResNet_101(img_shape, NUM_CLASSES, WEIGHT_DECAY)

        data_aug_params = {'horizontal_flip': True, 'width_shift_range': 0.125, 'height_shift_range': 0.125,
                           'fill_mode': 'constant', 'cval': 0.}
        early_stop_params = {'monitor': 'val_loss', 'mode': 'min', 'verbose': 1, 'patience': PATIENCE}

        ckpt_clbk = {'filepath': filepath + "/model_weights.h5",
                     'monitor': 'val_loss', 'mode': 'min', 'save_best_only': True, 'save_weights_only': True}
        train_params = {'epochs': EPOCHS, 'batch_size': BATCH_SIZE}

        #print(resnet.summary())
        resnet.compile(loss='categorical_crossentropy', optimizer='adam')

        resnet.fit(ckpt_clbk, train_params, x_train, y_train, x_val, y_val, data_aug_params,
                   early_stop_params, model_history)

        print("Get test accuracy:")
        loss, accuracy = resnet.evaluate(x_test, y_test)
        print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))

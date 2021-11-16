# Training procedure for CIFAR-10 using ResNet 110.
# ResNet model from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/4_Residual_Network/ResNet_keras.py

from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os
from constants import *
from context_fun import context_labels
from resnet_class.resnet101 import ResNet_101
from pathlib import Path

if __name__ == '__main__':
    # load data
    train_path = os.path.join("DataSets", "cifar5m_part0")
    valid_path = os.path.join("DataSets", "cifar5m_part1")

    for seed in SEEDS:
        filepath = os.path.join("model_weights", "resnet_101_blurdetail_context_nols_10_" + str(seed))
        Path(filepath).mkdir(parents=True, exist_ok=True)
        model_history = os.path.join(filepath, "model_history.p")

        #data_path = os.path.join("cifar5m", "train_set", "cifar5m_part0")

        train_file = "sample_3k_X.npy"
        train_label = "sample_3k_Y.npy"

        x_train = np.load(os.path.join(train_path, train_file))
        y_train = np.load(os.path.join(train_path, train_label))
        x_val = np.load(os.path.join(valid_path, train_file))
        y_val = np.load(os.path.join(valid_path, train_label))

        y_train = context_labels(y_train, "sample")
        y_val = context_labels(y_val, "sample")

        x_filter = []
        y_filter = []
        fil_valid_data = []
        fil_valid_label = []
        filter_list = ["blur", "detail"]  # , "edge_enhance", "smooth", "sharp"]

        for filter in filter_list:
            filter_file = filter + "_1k_X.npy"
            filter_label = filter + "_1k_Y.npy"
            fil_data = np.load(os.path.join(valid_path, filter_file))
            fil_label = np.load(os.path.join(valid_path, filter_label))
            #fil_label = context_labels(fil_label, filter, LABEL_SMOOTH)

            fil_data, fil_vdata, fil_label, fil_vlabel = train_test_split(fil_data, fil_label, test_size=500,
                                                                          random_state=seed)
            fil_vlabel = context_labels(fil_vlabel, filter, LABEL_SMOOTH)

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
                temp_y = context_labels(temp_y, filter, LABEL_SMOOTH)
                # ("temp_y shape: ", temp_y.shape)
                y_filter.append(temp_y)
                # print("Y_blur shape: ", np.array(y_blur).shape)


        x_filter = np.concatenate(x_filter)
        y_filter = np.concatenate(y_filter)
        fil_valid_data = np.concatenate(fil_valid_data)
        fil_valid_label = np.concatenate(fil_valid_label)

        print("Shape before reshape y_filter: ", y_filter.shape)
        y_filter.reshape(-1, y_filter.shape[-1])
        print("Shape after reshape y_filter: ", y_filter.shape)
        fil_valid_label.reshape(-1, fil_valid_label.shape[-1])

        print("Data shape before split: ", y_train.shape)

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

        #y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
        #y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
        #y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)

        print("x_train shape: ", x_train.shape)
        print("y_train shape: ", y_train.shape)

        img_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
        resnet = ResNet_101(img_shape, NUM_CLASSES, WEIGHT_DECAY, contexts=CONTEXTS)

        data_aug_params = {'horizontal_flip': True, 'width_shift_range': 0.125, 'height_shift_range': 0.125,
                           'fill_mode': 'constant', 'cval': 0.}
        early_stop_params = {'monitor': 'val_loss', 'mode': 'min', 'verbose': 1, 'patience': PATIENCE}

        ckpt_clbk = {'filepath': filepath + "/model_weights.h5",
                     'monitor': 'val_loss', 'mode': 'min', 'save_best_only': True, 'save_weights_only': True}
        train_params = {'epochs': EPOCHS, 'batch_size': BATCH_SIZE}


        #print(resnet.summary())
        resnet.compile(loss='categorical_crossentropy', optimizer='adam')

        resnet.fit(ckpt_clbk, train_params, x_train, [y_train[:, :10], y_train[:, 10:]],
                   x_val, [y_val[:, :10], y_val[:, 10:]], data_aug_params,
                   early_stop_params, model_history)

        print("Get test accuracy:")
        loss, accuracy = resnet.evaluate(x_test, y_test)
        print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))

# Load in model weights and evaluate its goodness (ECE, MCE, error) also saves logits.
# ResNet model from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/4_Residual_Network/ResNet_keras.py

from tensorflow import keras
import numpy as np
from resnet_class.resnet101 import ResNet_101
from sklearn.model_selection import train_test_split
import pandas as pd

# Imports to get "utility" package
import os

# sys.path.append(path.dirname(path.dirname(path.abspath("utility"))))
from calibration.evaluation import evaluate_model
from constants import *

if __name__ == '__main__':
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(os.path.join("results", "blur_detail_trained_seed_context.xlsx"), engine='xlsxwriter')

    for seed in SEEDS:
        #weights_file = os.path.join("model_weights", "resnet_110_10_blurdetail_u_" + str(seed) + ".h5")

        checkpoint_filepath = os.path.join("model_weights", "resnet_101_blurdetail_context_" + str(seed),
                                           "weights-blurdetail_context.h5")
        # load data
        results = pd.DataFrame(columns=["Filter", "Accuracy", "ECE", "MCE"])

        print("Cifar-5m evaluation")

        data_path = os.path.join("DataSets", "cifar5m_part2")

        filter_list = ["sample", "blur", "detail", "edge_enhance", "smooth", "sharp"]
        for fil in filter_list:
            # results_classwise = pd.DataFrame(columns=["Class", "Accuracy", "ECE", "MCE"])
            data_file = fil + "_1k_X.npy"
            data_label = fil + "_1k_Y.npy"

            x_test = np.load(os.path.join(data_path, data_file))
            y_test = np.load(os.path.join(data_path, data_label))
            # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

            # color preprocessing - using precalculated means and std-s
            if fil in ["blur", "detail", "edge_enhance", "smooth", "sharp"]:
                _, x_test, _, y_test = train_test_split(x_test, y_test, test_size=1000,
                                                            random_state=seed)  # random_state = seed
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.1,
                                                            random_state=seed)  # random_state = seed

            # build network
            img_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
            model = ResNet_101(img_shape, NUM_CLASSES, WEIGHT_DECAY, contexts=CONTEXTS)
            # model.load_weights(weights_file)

            # loss, accuracy = resnet.evaluate(x_test, y_test, verbose=0)
            # print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
            print("x_test shape: ", x_test.shape)
            print("y_test shape: ", y_test.shape)

            print("Evaluation metric for filter: {a}".format(a=fil))
            res = evaluate_model(checkpoint_filepath, x_test, y_test, bins=15, verbose=True,
                                 pickle_file="probs_resnet101_blurdetail_context_" + fil + "_" + str(seed), x_val=x_val,
                                 y_val=y_val, pickle_path="logits_context", contexts=CONTEXTS)

            res = {"Filter": fil, "Accuracy": res[0], "ECE": res[1], "MCE": res[2]}
            results = results.append(res, ignore_index=True)
            #               pickle_file="probs_resnet110_c10", x_val=x_val, y_val=y_val)
            '''
            for cl in range(10):
                class_y = np.load(os.path.join(data_path, data_label))
                class_x = x_test[class_y == cl]
                class_y = np.full(class_x.shape[0], cl, dtype=int)
                class_y = keras.utils.to_categorical(class_y, num_classes10)

                print("Evaluation metric for filter: {a} and class: {b}".format(a=fil, b= cl))
                class_res = evaluate_model(model, weights_file, class_x, class_y, bins=15, verbose=True)

                class_res = {"Class": cl, "Accuracy": class_res[0], "ECE": class_res[1], "MCE": class_res[2]}
                results_classwise = results_classwise.append(class_res, ignore_index=True)

            # Convert the dataframe to an XlsxWriter Excel object.
            print(results_classwise)
            results_classwise.to_excel(writer, sheet_name=fil, index=False)
            '''

        print(results)
        results.to_excel(writer, sheet_name="blurdetail_" + str(seed), index=False)
    writer.save()

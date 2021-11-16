import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pickle, os


class ResNet_101:
    def __init__(self, img_shape, num_classes, weight_decay,
                 last_layer_activation="softmax", contexts=False):
        if last_layer_activation == "linear":
            self.trainable = False
        else:
            self.trainable = True

        img_input = Input(shape=img_shape)
        img_input = tf.keras.applications.resnet.preprocess_input(img_input)

        self.resnet = ResNet101(include_top=False, weights='imagenet', #input_tensor=img_input,
                                input_shape=img_shape, pooling='avg',
                                classes=num_classes)

        i = self.resnet.input
        o = Dense(num_classes, activation=last_layer_activation, name="Prediction", kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(weight_decay))(self.resnet.layers[-1].output)
        if contexts:
            o_c = Dense(contexts, activation=last_layer_activation, name="Context", kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(weight_decay))(self.resnet.layers[-1].output)
            #o = Concatenate()([o, o_c])
            self.resnet = tf.keras.models.Model(inputs=i, outputs=[o, o_c])
        else:
            self.resnet = tf.keras.models.Model(inputs=i, outputs=[o])



    def summary(self):
        print(self.resnet.summary())

    def compile(self, loss, optimizer, metrics='accuracy'):
        if self.trainable:
            self.resnet.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        else:
            print("Model is not trainable. It must be used for prediction or calibration")

    def fit(self, ckpt_clbk, train_params, x_train, y_train, x_valid, y_valid, data_augment_params,
            early_stop_params=None, model_history=None):
        """
        datagen = ImageDataGenerator(horizontal_flip=data_augment_params['horizontal_flip'],
                                     width_shift_range=data_augment_params['width_shift_range'],
                                     height_shift_range=data_augment_params['height_shift_range'],
                                     fill_mode=data_augment_params['fill_mode'],
                                     cval=data_augment_params['cval'])
        """
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_clbk['filepath'],
                                                                       monitor=ckpt_clbk['monitor'],
                                                                       mode=ckpt_clbk['mode'],
                                                                       save_best_only=ckpt_clbk['save_best_only'],
                                                                       save_weights_only=ckpt_clbk['save_weights_only'])
        callbacks = [model_checkpoint_callback]

        if early_stop_params is not None:
            es = EarlyStopping(monitor=early_stop_params['monitor'], mode=early_stop_params['mode'],
                               verbose=early_stop_params['verbose'], patience=early_stop_params['patience'])
            callbacks.append(es)

        # start training
        self.hist = self.resnet.fit(x_train, y_train, batch_size=train_params['batch_size'],
                                    epochs=train_params['epochs'],
                                    callbacks=callbacks,
                                    validation_data=(x_valid, y_valid))

        if model_history is not None:
            print("Pickle models history")
            with open(model_history, 'wb') as f:
                pickle.dump(self.hist.history, f)

    def evaluate(self, x_test, y_test, verbose=0):
        loss, accuracy = self.resnet.evaluate(x_test, y_test, verbose=verbose)
        return (loss, accuracy)

    def predict(self, x_test, verbose=1):
        return self.resnet.predict(x_test, verbose=verbose)


    def load_weights(self, weights_file):
        self.resnet.load_weights(weights_file) #.expect_partial()

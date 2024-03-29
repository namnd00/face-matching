import pandas as pd
import numpy as np

import keras
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import *
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, Activation, BatchNormalization, Dense, Dropout, Flatten, add, \
    Lambda
import tensorflow as tf
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from clr_callback import CyclicLR
import random
import threading
from random import randint
import os

BATCH_SIZE = 32
EPOCHS = 100
NUMBER_OF_FOLDS = 5
NUMBER_OF_PARTS = 4
INPUT_DIM = 192 * 2
NUMBER_OF_CLASSES = 2
#
#
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

K.tensorflow_backend._get_available_gpus()

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


class ThreadSafeIterator:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
	A decorator that takes a generator function and makes it thread-safe.
	"""

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


@threadsafe_generator
def train_generator(xtrain_fold, ytrain_fold, train_size, batch_size):
    while True:
        xtrain_fold, ytrain_fold = shuffle(xtrain_fold, ytrain_fold)
        for start in range(0, train_size, batch_size):
            end = min(start + batch_size, train_size)
            x_batch = np.array([], dtype=np.float32).reshape(0, INPUT_DIM)
            for i in range(start, end, 1):
                x_batch = np.vstack((x_batch, xtrain_fold[i, randint(0, 99), :].reshape(1, INPUT_DIM)))
            y_batch = ytrain_fold[start:end, :]
            yield x_batch, y_batch


@threadsafe_generator
def valid_generator(xvalid_fold, yvalid_fold, valid_size, batch_size):
    while True:
        for start in range(0, valid_size, batch_size):
            end = min(start + batch_size, valid_size)
            x_batch = xvalid_fold[start:end, :]
            y_batch = yvalid_fold[start:end, :]
            yield x_batch, y_batch


def get_y_true(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(to_categorical(row['label'], num_classes=NUMBER_OF_CLASSES))
    return np.array(y_true)


def Model():
    model = Sequential()
    model.add(Dense(2048, input_dim=INPUT_DIM, kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(NUMBER_OF_CLASSES, kernel_initializer='uniform'))
    model.add(Activation('softmax'))
    return model


if __name__ == '__main__':
    test_df = pd.read_csv('test_refined.csv')
    train_df = pd.read_csv('train_refined.csv')

    xtrain = np.load('train_data.npy')
    xtrain_aug = np.load('train_aug_data.npy')
    ytrain = get_y_true(train_df)
    xtest = np.load('test_data.npy')
    # xtest_flip = np.load('test_flip_data.npy')

    if not os.path.exists('weights'):
        os.makedirs('weights')

    ptest = np.zeros((xtest.shape[0], NUMBER_OF_CLASSES), dtype=np.float64)
    # training_log = open('training_log.txt', 'w')
    loss_average = 0.0
    acc_average = 0.0
    for part in random.sample(range(10), NUMBER_OF_PARTS):
        for fold in range(NUMBER_OF_FOLDS):
            v_df = train_df.loc[train_df['rt%d' % part] == fold]
            vidxs = v_df.index.values.tolist()
            t_df = train_df.loc[~train_df.index.isin(v_df.index)]
            tidxs = t_df.index.values.tolist()
            print('**************Part %d    Fold %d**************' % (part, fold))

            xtrain_fold = xtrain_aug[tidxs, :, :]
            ytrain_fold = ytrain[tidxs, :]

            xvalid_fold = xtrain[vidxs, :]
            yvalid_fold = ytrain[vidxs, :]

            train_size = len(tidxs)
            valid_size = len(vidxs)
            train_steps = np.ceil(float(train_size) / float(BATCH_SIZE))
            valid_steps = np.ceil(float(valid_size) / float(BATCH_SIZE))
            print('TRAIN SIZE: %d VALID SIZE: %d' % (train_size, valid_size))

            WEIGHTS_BEST = 'weights/best_weight_part%d_fold%d.hdf5' % (part, fold)

            clr = CyclicLR(base_lr=1e-7, max_lr=1e-3, step_size=6 * train_steps, mode='exp_range', gamma=0.99994)
            early_stopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
            save_checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor='val_acc', verbose=1, save_weights_only=True,
                                              save_best_only=True, mode='max')
            callbacks = [save_checkpoint, early_stopping, clr]

            model = Model()
            model.summary()
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

            model.fit_generator(generator=train_generator(xtrain_fold, ytrain_fold, train_size, BATCH_SIZE),
                                steps_per_epoch=train_steps, epochs=EPOCHS, verbose=1,
                                validation_data=valid_generator(xvalid_fold, yvalid_fold, valid_size, BATCH_SIZE),
                                validation_steps=valid_steps, callbacks=callbacks)

            model.load_weights(WEIGHTS_BEST)

            ptest += model.predict(xtest, batch_size=BATCH_SIZE, verbose=1)
            # ptest += model.predict(xtest_flip, batch_size=BATCH_SIZE, verbose=1)

            score = model.evaluate(x=xvalid_fold, y=yvalid_fold, batch_size=BATCH_SIZE, verbose=1)
            loss_average += score[0]
            acc_average += score[1]
            with open('training_log.txt', 'a') as training_log:
                training_log.write('PART:%d FOLD:%d LOSS:%f ACC:%f\n' % (part, fold, score[0], score[1]))
            K.clear_session()

    ptest /= float(2 * NUMBER_OF_PARTS * NUMBER_OF_FOLDS)
    np.save('ptest.npy', ptest)

    loss_average /= float(NUMBER_OF_PARTS * NUMBER_OF_FOLDS)
    acc_average /= float(NUMBER_OF_PARTS * NUMBER_OF_FOLDS)

    with open('training_log.txt', 'a') as training_log:
        training_log.write('AVERAGE LOSS:%f ACC:%f\n' % (loss_average, acc_average))

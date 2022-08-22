
import keras
from keras import backend as K
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
# from keras.layers import Merge, Input, merge
from keras.layers import  Input, merge

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import h5py
from keras.layers.merge import Multiply
from keras.layers import Activation, Dense


def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])


def model1(n_time, n_freq,  num_classes):
    input_logmel = Input(shape=(n_time, n_freq))
    a1 = Reshape((n_time, n_freq, 1))(input_logmel)

    cnn1 = block(a1)
    cnn1 = block(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)

    cnn1 = block(cnn1)
    cnn1 = block(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)

    cnn1 = block(cnn1)
    cnn1 = block(cnn1)
    cnn1 = MaxPooling2D(pool_size=(2, 2))(cnn1)

    cnn1 = block(cnn1)
    cnn3 = block(cnn1)
    cnn3 = MaxPooling2D(pool_size=(1, 2))(cnn1)

    cnnout = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(cnn3)
    cnnout = MaxPooling2D(pool_size=(1, 4))(cnnout)

    cnnout = Reshape((47, 256))(cnnout)  # Time step is downsampled to 30.

    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(cnnout)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(cnnout)
    out = Multiply()([rnnout, rnnout_gate])

    out = TimeDistributed(Dense(num_classes, activation='sigmoid'))(out)
    out = Lambda(lambda x: K.mean(x, axis=1), output_shape=(num_classes,))(out)
    out = Activation('sigmoid')(out)

    model = Model(input_logmel, out)
    return model


def modeltest(n_time, n_freq,  num_classes):
    input_logmel = Input(shape=(n_time, n_freq))
    a1 = Reshape((n_time, n_freq, 1))(input_logmel)

    cnn1 = block(a1)
    cnn1 = MaxPooling2D(pool_size=(3,3 ))(cnn1)
    cnn1 = Conv2D( 1, ( 3, 3), padding="same", activation="linear", use_bias=False)(cnn1)
    cnn1 = BatchNormalization(axis=-1)(cnn1)
    cnnout = Reshape((21, 7))(cnn1)  # Time step is downsampled to 30.
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(cnnout)

    # fulllay = Flatten(input_shape=[21, 256])(rnnout)
    fulllay = Reshape((21 * 256,))(rnnout)

    fulllay = Dense(128 , activation='linear')(fulllay)

    out = Dense(units =num_classes, activation = 'softmax')(fulllay)

    model = Model(input_logmel, out)
    return model


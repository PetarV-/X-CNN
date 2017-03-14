'''
This will attempt to reimplement the FitNet4 network
as described by Romero et al. (2014)
'''

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Dropout, merge, MaxoutDense
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from utils.preprocess import get_cifar

batch_size = 128
nb_classes = 10
nb_epoch = 230
data_augmentation = True

# show the summary?
show_summary = True

# save the weights after training?
save_weights = True
weights_file = 'cifar10_fitnet.h5'

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = get_cifar(p=1.0, append_test=False, use_c10=True)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

inputYUV = Input(shape=(3, 32, 32))
inputNorm = BatchNormalization(axis=1)(inputYUV)

input_drop = Dropout(0.2)(inputNorm)

# This is a single convolutional maxout layer.
h0_conv_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(input_drop)
h0_conv_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(input_drop)
h0_conv = merge([h0_conv_a, h0_conv_b], mode='max', concat_axis=1)
h0_conv = BatchNormalization(axis=1)(h0_conv)

h1_conv_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv)
h1_conv_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv)
h1_conv = merge([h1_conv_a, h1_conv_b], mode='max', concat_axis=1)
h1_conv = BatchNormalization(axis=1)(h1_conv)

h2_conv_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv)
h2_conv_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv)
h2_conv = merge([h2_conv_a, h2_conv_b], mode='max', concat_axis=1)
h2_conv = BatchNormalization(axis=1)(h2_conv)

h3_conv_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv)
h3_conv_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv)
h3_conv = merge([h3_conv_a, h3_conv_b], mode='max', concat_axis=1)
h3_conv = BatchNormalization(axis=1)(h3_conv)

h4_conv_a = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv)
h4_conv_b = Convolution2D(48, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv)
h4_conv = merge([h4_conv_a, h4_conv_b], mode='max', concat_axis=1)
h4_conv = BatchNormalization(axis=1)(h4_conv)

h4_pool = MaxPooling2D(pool_size=(2, 2))(h4_conv)
h4_drop = Dropout(0.2)(h4_pool)

h5_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h4_drop)
h5_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h4_drop)
h5_conv = merge([h5_conv_a, h5_conv_b], mode='max', concat_axis=1)
h5_conv = BatchNormalization(axis=1)(h5_conv)

h6_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv)
h6_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv)
h6_conv = merge([h6_conv_a, h6_conv_b], mode='max', concat_axis=1)
h6_conv = BatchNormalization(axis=1)(h6_conv)

h7_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv)
h7_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv)
h7_conv = merge([h7_conv_a, h7_conv_b], mode='max', concat_axis=1)
h7_conv = BatchNormalization(axis=1)(h7_conv)

h8_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv)
h8_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv)
h8_conv = merge([h8_conv_a, h8_conv_b], mode='max', concat_axis=1)
h8_conv = BatchNormalization(axis=1)(h8_conv)

h9_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv)
h9_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv)
h9_conv = merge([h9_conv_a, h9_conv_b], mode='max', concat_axis=1)
h9_conv = BatchNormalization(axis=1)(h9_conv)

h10_conv_a = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv)
h10_conv_b = Convolution2D(80, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv)
h10_conv = merge([h10_conv_a, h10_conv_b], mode='max', concat_axis=1)
h10_conv = BatchNormalization(axis=1)(h10_conv)

h10_pool = MaxPooling2D(pool_size=(2, 2))(h10_conv)
h10_drop = Dropout(0.2)(h10_pool)

h11_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h10_drop)
h11_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h10_drop)
h11_conv = merge([h11_conv_a, h11_conv_b], mode='max', concat_axis=1)
h11_conv = BatchNormalization(axis=1)(h11_conv)

h12_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv)
h12_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv)
h12_conv = merge([h12_conv_a, h12_conv_b], mode='max', concat_axis=1)
h12_conv = BatchNormalization(axis=1)(h12_conv)

h13_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv)
h13_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv)
h13_conv = merge([h13_conv_a, h13_conv_b], mode='max', concat_axis=1)
h13_conv = BatchNormalization(axis=1)(h13_conv)

h14_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv)
h14_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv)
h14_conv = merge([h14_conv_a, h14_conv_b], mode='max', concat_axis=1)
h14_conv = BatchNormalization(axis=1)(h14_conv)

h15_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv)
h15_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv)
h15_conv = merge([h15_conv_a, h15_conv_b], mode='max', concat_axis=1)
h15_conv = BatchNormalization(axis=1)(h15_conv)

h16_conv_a = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv)
h16_conv_b = Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv)
h16_conv = merge([h16_conv_a, h16_conv_b], mode='max', concat_axis=1)
h16_conv = BatchNormalization(axis=1)(h16_conv)

h16_pool = MaxPooling2D(pool_size=(8, 8))(h16_conv)
h16_drop = Dropout(0.2)(h16_pool)

h16 = Flatten()(h16_drop)
h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
h17 = BatchNormalization(axis=1)(h17)
h17_drop = Dropout(0.2)(h17)
out = Dense(nb_classes, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17_drop)

model = Model(input=inputYUV, output=out)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if show_summary:
    print(model.summary())

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              verbose=2)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                        verbose=2)


if save_weights:
    model.save_weights(weights_file)


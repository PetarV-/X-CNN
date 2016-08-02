'''
This is the multilayer variant of the CIFAR-10 All-CNN-C (differently placed merge, in line with 4L)
'''

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Dropout, merge, Lambda
from keras.layers import Convolution2D, AveragePooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from utils.preprocess import get_cifar

batch_size = 64
nb_classes = 10
nb_epoch = 350
alpha = 0.001 # weight decay parameter
data_augmentation = True

# plot the model?
plot_model = True
show_shapes = True
plot_file = 'cifar10_acn_multi2.png'

# show the summary?
show_summary = True

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = get_cifar(p=1.0, append_test=False, use_c10=True)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

inputYUV = Input(shape=(3, 32, 32))

# To simplify the data augmentation, I delay slicing until this point.
# Not sure if there is a better way to handle it. ---Petar
inputY = Lambda(lambda x: x[:,0:1,:,:], output_shape=(1, 32, 32))(inputYUV)
inputU = Lambda(lambda x: x[:,1:2,:,:], output_shape=(1, 32, 32))(inputYUV)
inputV = Lambda(lambda x: x[:,2:3,:,:], output_shape=(1, 32, 32))(inputYUV)

inputY_drop = Dropout(0.2)(inputY)
inputU_drop = Dropout(0.2)(inputU)
inputV_drop = Dropout(0.2)(inputV)

h0_conv_Y = Convolution2D(48, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(inputY_drop)
h0_conv_U = Convolution2D(24, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(inputU_drop)
h0_conv_V = Convolution2D(24, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(inputV_drop)

h1_conv_Y = Convolution2D(48, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h0_conv_Y)
h1_conv_U = Convolution2D(24, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h0_conv_U)
h1_conv_V = Convolution2D(24, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h0_conv_V)

# "Pooling" convolutions 1
h2_conv_Y = Convolution2D(48, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha), subsample=(2, 2))(h1_conv_Y)
h2_conv_U = Convolution2D(24, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha), subsample=(2, 2))(h1_conv_U)
h2_conv_V = Convolution2D(24, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha), subsample=(2, 2))(h1_conv_V)

h2_drop_Y = Dropout(0.5)(h2_conv_Y)
h2_drop_U = Dropout(0.5)(h2_conv_U)
h2_drop_V = Dropout(0.5)(h2_conv_V)

# Interlayer connections Y <-> U, Y <-> V
Y_to_UV = Convolution2D(48, 1, 1, border_mode='same', activation='relu')(h2_drop_Y)
U_to_Y = Convolution2D(24, 1, 1, border_mode='same', activation='relu')(h2_drop_U)
V_to_Y = Convolution2D(24, 1, 1, border_mode='same', activation='relu')(h2_drop_V)

h2_Y = merge([h2_drop_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=1)
h2_U = merge([h2_drop_U, Y_to_UV], mode='concat', concat_axis=1)
h2_V = merge([h2_drop_V, Y_to_UV], mode='concat', concat_axis=1)

h3_conv_Y = Convolution2D(96, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h2_Y)
h3_conv_U = Convolution2D(48, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h2_U)
h3_conv_V = Convolution2D(48, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h2_V)

h4_conv_Y = Convolution2D(96, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h3_conv_Y)
h4_conv_U = Convolution2D(48, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h3_conv_U)
h4_conv_V = Convolution2D(48, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h3_conv_V)

# "Pooling" convolution 2
h5_conv_Y = Convolution2D(96, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha), subsample=(2, 2))(h4_conv_Y)
h5_conv_U = Convolution2D(48, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha), subsample=(2, 2))(h4_conv_U)
h5_conv_V = Convolution2D(48, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha), subsample=(2, 2))(h4_conv_V)

# In this version, interlayer connections end here (equivalent to 4L)
h5_conv = merge([h5_conv_Y, h5_conv_U, h5_conv_V], mode='concat', concat_axis=1)
h5_drop = Dropout(0.5)(h5_conv)

# Some more convolutions
h6_conv = Convolution2D(192, 3, 3, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h5_drop)
h7_conv = Convolution2D(192, 1, 1, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h6_conv)
h8_conv = Convolution2D(nb_classes, 1, 1, border_mode='same', activation='relu', W_regularizer=l2(alpha))(h7_conv)

# Now average and softmax
h9_conv = AveragePooling2D(pool_size=(8, 8))(h8_conv)
h9_flat = Flatten()(h9_conv)
out = Activation('softmax')(h9_flat)

model = Model(input=inputYUV, output=out)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

if show_summary:
    print(model.summary())

if plot_model:
    plot(model, show_shapes=show_shapes, to_file=plot_file)

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

'''
This is the multilayer variant of the CIFAR-10 maxout network
'''

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Activation, Flatten, merge, MaxoutDense
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from utils.preprocess import get_cifar

batch_size = 128
nb_classes = 10
nb_epoch = 474
data_augmentation = True

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = get_cifar(p=1.0, append_test=False, use_c10=True)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

inputYUV = Input(shape=(3, 32, 32))

# To simplify the data augmentation, I delay slicing until this point.
# Not sure if there is a better way to handle it. ---Petar
inputY = Lambda(lambda x: x[:,0:1,:,:], output_shape=(1, 32, 32))(inputYUV)
inputU = Lambda(lambda x: x[:,1:2,:,:], output_shape=(1, 32, 32))(inputYUV)
inputV = Lambda(lambda x: x[:,2:3,:,:], output_shape=(1, 32, 32))(inputYUV)

# The first Maxout-Conv layer, split into the three layers
h0_pad_Y = ZeroPadding2D((4, 4))(inputY)
h0_conv_a_Y = Convolution2D(48, 8, 8, border_mode='valid')(h0_pad_Y)
h0_conv_b_Y = Convolution2D(48, 8, 8, border_mode='valid')(h0_pad_Y)
h0_conv_Y = merge([h0_conv_a_Y, h0_conv_b_Y], mode='max', concat_axis=1)
h0_pool_Y = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(h0_conv_Y)

h0_pad_U = ZeroPadding2D((4, 4))(inputU)
h0_conv_a_U = Convolution2D(24, 8, 8, border_mode='valid')(h0_pad_U)
h0_conv_b_U = Convolution2D(24, 8, 8, border_mode='valid')(h0_pad_U)
h0_conv_U = merge([h0_conv_a_U, h0_conv_b_U], mode='max', concat_axis=1)
h0_pool_U = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(h0_conv_U)

h0_pad_V = ZeroPadding2D((4, 4))(inputV)
h0_conv_a_V = Convolution2D(24, 8, 8, border_mode='valid')(h0_pad_V)
h0_conv_b_V = Convolution2D(24, 8, 8, border_mode='valid')(h0_pad_V)
h0_conv_V = merge([h0_conv_a_V, h0_conv_b_V], mode='max', concat_axis=1)
h0_pool_V = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(h0_conv_V)

# The second Maxout-Conv layer, split into three layers
h1_pad_Y = ZeroPadding2D((3, 3))(h0_pool_Y)
h1_conv_a_Y = Convolution2D(96, 8, 8, border_mode='valid')(h1_pad_Y)
h1_conv_b_Y = Convolution2D(96, 8, 8, border_mode='valid')(h1_pad_Y)
h1_conv_Y = merge([h1_conv_a_Y, h1_conv_b_Y], mode='max', concat_axis=1)
h1_pool_Y = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(h1_conv_Y)

h1_pad_U = ZeroPadding2D((3, 3))(h0_pool_U)
h1_conv_a_U = Convolution2D(48, 8, 8, border_mode='valid')(h1_pad_U)
h1_conv_b_U = Convolution2D(48, 8, 8, border_mode='valid')(h1_pad_U)
h1_conv_U = merge([h1_conv_a_U, h1_conv_b_U], mode='max', concat_axis=1)
h1_pool_U = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(h1_conv_U)

h1_pad_V = ZeroPadding2D((3, 3))(h0_pool_V)
h1_conv_a_V = Convolution2D(48, 8, 8, border_mode='valid')(h1_pad_V)
h1_conv_b_V = Convolution2D(48, 8, 8, border_mode='valid')(h1_pad_V)
h1_conv_V = merge([h1_conv_a_V, h1_conv_b_V], mode='max', concat_axis=1)
h1_pool_V = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(h1_conv_V)

# Interlayer connections Y <-> U, Y <-> V; use maxout
Y_to_UV_a = Convolution2D(96, 1, 1, border_mode='same')(h1_pool_Y)
Y_to_UV_b = Convolution2D(96, 1, 1, border_mode='same')(h1_pool_Y)
Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=1)

U_to_Y_a = Convolution2D(48, 1, 1, border_mode='same')(h1_pool_U)
U_to_Y_b = Convolution2D(48, 1, 1, border_mode='same')(h1_pool_U)
U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=1)

V_to_Y_a = Convolution2D(48, 1, 1, border_mode='same')(h1_pool_V)
V_to_Y_b = Convolution2D(48, 1, 1, border_mode='same')(h1_pool_V)
V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=1)

# concatenate intra- and inter-layer values
Y_concat = merge([h1_pool_Y, U_to_Y, V_to_Y], mode='concat', concat_axis=1)
U_concat = merge([h1_pool_U, Y_to_UV], mode='concat', concat_axis=1)
V_concat = merge([h1_pool_V, Y_to_UV], mode='concat', concat_axis=1)

# The third Maxout-Conv layer, split into three layers
h2_pad_Y = ZeroPadding2D((3, 3))(Y_concat)
h2_conv_a_Y = Convolution2D(96, 5, 5, border_mode='valid')(h2_pad_Y)
h2_conv_b_Y = Convolution2D(96, 5, 5, border_mode='valid')(h2_pad_Y)
h2_conv_Y = merge([h2_conv_a_Y, h2_conv_b_Y], mode='max', concat_axis=1)
h2_pool_Y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(h2_conv_Y)

h2_pad_U = ZeroPadding2D((3, 3))(U_concat)
h2_conv_a_U = Convolution2D(48, 5, 5, border_mode='valid')(h2_pad_U)
h2_conv_b_U = Convolution2D(48, 5, 5, border_mode='valid')(h2_pad_U)
h2_conv_U = merge([h2_conv_a_U, h2_conv_b_U], mode='max', concat_axis=1)
h2_pool_U = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(h2_conv_U)

h2_pad_V = ZeroPadding2D((3, 3))(V_concat)
h2_conv_a_V = Convolution2D(48, 5, 5, border_mode='valid')(h2_pad_V)
h2_conv_b_V = Convolution2D(48, 5, 5, border_mode='valid')(h2_pad_V)
h2_conv_V = merge([h2_conv_a_V, h2_conv_b_V], mode='max', concat_axis=1)
h2_pool_V = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(h2_conv_V)

# Merge and flatten the three layers
h2_concat = merge([h2_pool_Y, h2_pool_U, h2_pool_V], mode='concat', concat_axis=1)
h2_flat = Flatten()(h2_concat)

# Now the more conventional layers...
h3 = MaxoutDense(500, nb_feature=5)(h2_flat)
out = Dense(nb_classes)(h3)
y = Activation('softmax')(out) 

model = Model(input=inputYUV, output=out)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
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
                        validation_data=(X_test, Y_test))

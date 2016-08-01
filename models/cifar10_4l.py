from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, merge, Input, Lambda
from keras.utils.visualize_util import plot
from utils.preprocess import get_cifar

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# plot the model?
plot_model = True
show_shapes = True
plot_file = 'cifar10_4l.png'

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

#cross-connections between two conv layers, Y is the middle layer, while U and V are side layers.

inputYUV = Input(shape=(3, 32, 32))

# To simplify the data augmentation, I delay slicing until this point.
# Not sure if there is a better way to handle it. ---Petar
inputY = Lambda(lambda x: x[:,0:1,:,:], output_shape=(1, 32, 32))(inputYUV)
inputU = Lambda(lambda x: x[:,1:2,:,:], output_shape=(1, 32, 32))(inputYUV)
inputV = Lambda(lambda x: x[:,2:3,:,:], output_shape=(1, 32, 32))(inputYUV)

convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

poolY = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(convY)
poolU = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(convU)
poolV = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(convV)

poolY = Dropout(0.25)(poolY)
poolU = Dropout(0.25)(poolU)
poolV = Dropout(0.25)(poolV)

U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
Y_to_UV = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)

Ymap = merge([poolY,U_to_Y,V_to_Y], mode='concat', concat_axis=1)
Umap = merge([poolU,Y_to_UV], mode='concat', concat_axis=1)
Vmap = merge([poolV,Y_to_UV], mode='concat', concat_axis=1)



convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)


convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

poolY = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(convY)
poolU = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(convU)
poolV = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(convV)

poolY = Dropout(0.25)(poolY)
poolU = Dropout(0.25)(poolU)
poolV = Dropout(0.25)(poolV)


concatenate_map=merge([poolY,poolU,poolV], mode='concat', concat_axis=1)

reshape=Flatten()(concatenate_map)
fc=Dense(512, activation='relu')(reshape)
fc=Dropout(0.5)(fc)
out=Dense(nb_classes, activation='softmax')(fc)

model = Model(input=inputYUV, output=out)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
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

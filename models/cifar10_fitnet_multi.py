'''
This will implement the multilayer variant of the FitNet4 network
'''

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Dropout, merge, MaxoutDense
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.visualize_util import plot
from utils.preprocess import get_cifar

batch_size = 128
nb_classes = 10
nb_epoch = 230
data_augmentation = True

# plot the model?
plot_model = True
show_shapes = True
plot_file = 'cifar10_fitnet_multi.png'

# show the summary?
show_summary = True

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = get_cifar(p=1.0, append_test=False, use_c10=True)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

inputYUV = Input(shape=(3, 32, 32))
inputNorm = BatchNormalization(axis=1)(inputYUV)

inputY = Lambda(lambda x: x[:,0:1,:,:], output_shape=(1, 32, 32))(inputNorm)
inputU = Lambda(lambda x: x[:,1:2,:,:], output_shape=(1, 32, 32))(inputNorm)
inputV = Lambda(lambda x: x[:,2:3,:,:], output_shape=(1, 32, 32))(inputNorm)

inputY_drop = Dropout(0.2)(inputY)
inputU_drop = Dropout(0.2)(inputU)
inputV_drop = Dropout(0.2)(inputV)

h0_conv_Y_a = Convolution2D(16, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
h0_conv_Y_b = Convolution2D(16, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputY)
h0_conv_Y = merge([h0_conv_Y_a, h0_conv_Y_b], mode='max', concat_axis=1)
h0_conv_Y = BatchNormalization(axis=1)(h0_conv_Y)

h0_conv_U_a = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
h0_conv_U_b = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputU)
h0_conv_U = merge([h0_conv_U_a, h0_conv_U_b], mode='max', concat_axis=1)
h0_conv_U = BatchNormalization(axis=1)(h0_conv_U)

h0_conv_V_a = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
h0_conv_V_b = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(inputV)
h0_conv_V = merge([h0_conv_V_a, h0_conv_V_b], mode='max', concat_axis=1)
h0_conv_V = BatchNormalization(axis=1)(h0_conv_V)

h1_conv_Y_a = Convolution2D(16, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_Y)
h1_conv_Y_b = Convolution2D(16, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_Y)
h1_conv_Y = merge([h1_conv_Y_a, h1_conv_Y_b], mode='max', concat_axis=1)
h1_conv_Y = BatchNormalization(axis=1)(h1_conv_Y)

h1_conv_U_a = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_U)
h1_conv_U_b = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_U)
h1_conv_U = merge([h1_conv_U_a, h1_conv_U_b], mode='max', concat_axis=1)
h1_conv_U = BatchNormalization(axis=1)(h1_conv_U)

h1_conv_V_a = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_V)
h1_conv_V_b = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h0_conv_V)
h1_conv_V = merge([h1_conv_V_a, h1_conv_V_b], mode='max', concat_axis=1)
h1_conv_V = BatchNormalization(axis=1)(h1_conv_V)

h2_conv_Y_a = Convolution2D(16, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_Y)
h2_conv_Y_b = Convolution2D(16, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_Y)
h2_conv_Y = merge([h2_conv_Y_a, h2_conv_Y_b], mode='max', concat_axis=1)
h2_conv_Y = BatchNormalization(axis=1)(h2_conv_Y)

h2_conv_U_a = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_U)
h2_conv_U_b = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_U)
h2_conv_U = merge([h2_conv_U_a, h2_conv_U_b], mode='max', concat_axis=1)
h2_conv_U = BatchNormalization(axis=1)(h2_conv_U)

h2_conv_V_a = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_V)
h2_conv_V_b = Convolution2D(8, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h1_conv_V)
h2_conv_V = merge([h2_conv_V_a, h2_conv_V_b], mode='max', concat_axis=1)
h2_conv_V = BatchNormalization(axis=1)(h2_conv_V)

h3_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_Y)
h3_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_Y)
h3_conv_Y = merge([h3_conv_Y_a, h3_conv_Y_b], mode='max', concat_axis=1)
h3_conv_Y = BatchNormalization(axis=1)(h3_conv_Y)

h3_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_U)
h3_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_U)
h3_conv_U = merge([h3_conv_U_a, h3_conv_U_b], mode='max', concat_axis=1)
h3_conv_U = BatchNormalization(axis=1)(h3_conv_U)

h3_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_V)
h3_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h2_conv_V)
h3_conv_V = merge([h3_conv_V_a, h3_conv_V_b], mode='max', concat_axis=1)
h3_conv_V = BatchNormalization(axis=1)(h3_conv_V)

h4_conv_Y_a = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_Y)
h4_conv_Y_b = Convolution2D(24, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_Y)
h4_conv_Y = merge([h4_conv_Y_a, h4_conv_Y_b], mode='max', concat_axis=1)
h4_conv_Y = BatchNormalization(axis=1)(h3_conv_Y)

h4_conv_U_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_U)
h4_conv_U_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_U)
h4_conv_U = merge([h4_conv_U_a, h4_conv_U_b], mode='max', concat_axis=1)
h4_conv_U = BatchNormalization(axis=1)(h3_conv_U)

h4_conv_V_a = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_V)
h4_conv_V_b = Convolution2D(12, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h3_conv_V)
h4_conv_V = merge([h4_conv_V_a, h4_conv_V_b], mode='max', concat_axis=1)
h4_conv_V = BatchNormalization(axis=1)(h4_conv_V)

poolY = MaxPooling2D(pool_size=(2, 2))(h4_conv_Y)
poolU = MaxPooling2D(pool_size=(2, 2))(h4_conv_U)
poolV = MaxPooling2D(pool_size=(2, 2))(h4_conv_V)

poolY = Dropout(0.2)(poolY)
poolU = Dropout(0.2)(poolU)
poolV = Dropout(0.2)(poolV)

# Cross connections: Y <-> U, Y <-> V
Y_to_UV_a = Convolution2D(24, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
Y_to_UV_b = Convolution2D(24, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=1)
Y_to_UV = BatchNormalization(axis=1)(Y_to_UV)

U_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
U_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=1)
U_to_Y = BatchNormalization(axis=1)(U_to_Y)

V_to_Y_a = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
V_to_Y_b = Convolution2D(12, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=1)
V_to_Y = BatchNormalization(axis=1)(V_to_Y)

Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=1)
Umap = merge([poolU, Y_to_UV], mode='concat', concat_axis=1)
Vmap = merge([poolV, Y_to_UV], mode='concat', concat_axis=1)

h5_conv_Y_a = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
h5_conv_Y_b = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
h5_conv_Y = merge([h5_conv_Y_a, h5_conv_Y_b], mode='max', concat_axis=1)
h5_conv_Y = BatchNormalization(axis=1)(h5_conv_Y)

h5_conv_U_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
h5_conv_U_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
h5_conv_U = merge([h5_conv_U_a, h5_conv_U_b], mode='max', concat_axis=1)
h5_conv_U = BatchNormalization(axis=1)(h5_conv_U)

h5_conv_V_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
h5_conv_V_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
h5_conv_V = merge([h5_conv_V_a, h5_conv_V_b], mode='max', concat_axis=1)
h5_conv_V = BatchNormalization(axis=1)(h5_conv_V)

h6_conv_Y_a = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_Y)
h6_conv_Y_b = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_Y)
h6_conv_Y = merge([h6_conv_Y_a, h6_conv_Y_b], mode='max', concat_axis=1)
h6_conv_Y = BatchNormalization(axis=1)(h6_conv_Y)

h6_conv_U_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_U)
h6_conv_U_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_U)
h6_conv_U = merge([h6_conv_U_a, h6_conv_U_b], mode='max', concat_axis=1)
h6_conv_U = BatchNormalization(axis=1)(h6_conv_U)

h6_conv_V_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_V)
h6_conv_V_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h5_conv_V)
h6_conv_V = merge([h6_conv_V_a, h6_conv_V_b], mode='max', concat_axis=1)
h6_conv_V = BatchNormalization(axis=1)(h6_conv_V)

h7_conv_Y_a = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_Y)
h7_conv_Y_b = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_Y)
h7_conv_Y = merge([h7_conv_Y_a, h7_conv_Y_b], mode='max', concat_axis=1)
h7_conv_Y = BatchNormalization(axis=1)(h7_conv_Y)

h7_conv_U_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_U)
h7_conv_U_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_U)
h7_conv_U = merge([h7_conv_U_a, h7_conv_U_b], mode='max', concat_axis=1)
h7_conv_U = BatchNormalization(axis=1)(h7_conv_U)

h7_conv_V_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_V)
h7_conv_V_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h6_conv_V)
h7_conv_V = merge([h7_conv_V_a, h7_conv_V_b], mode='max', concat_axis=1)
h7_conv_V = BatchNormalization(axis=1)(h7_conv_V)

h8_conv_Y_a = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_Y)
h8_conv_Y_b = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_Y)
h8_conv_Y = merge([h8_conv_Y_a, h8_conv_Y_b], mode='max', concat_axis=1)
h8_conv_Y = BatchNormalization(axis=1)(h8_conv_Y)

h8_conv_U_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_U)
h8_conv_U_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_U)
h8_conv_U = merge([h8_conv_U_a, h8_conv_U_b], mode='max', concat_axis=1)
h8_conv_U = BatchNormalization(axis=1)(h8_conv_U)

h8_conv_V_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_V)
h8_conv_V_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h7_conv_V)
h8_conv_V = merge([h8_conv_V_a, h8_conv_V_b], mode='max', concat_axis=1)
h8_conv_V = BatchNormalization(axis=1)(h8_conv_V)

h9_conv_Y_a = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_Y)
h9_conv_Y_b = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_Y)
h9_conv_Y = merge([h9_conv_Y_a, h9_conv_Y_b], mode='max', concat_axis=1)
h9_conv_Y = BatchNormalization(axis=1)(h9_conv_Y)

h9_conv_U_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_U)
h9_conv_U_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_U)
h9_conv_U = merge([h9_conv_U_a, h9_conv_U_b], mode='max', concat_axis=1)
h9_conv_U = BatchNormalization(axis=1)(h9_conv_U)

h9_conv_V_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_V)
h9_conv_V_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h8_conv_V)
h9_conv_V = merge([h9_conv_V_a, h9_conv_V_b], mode='max', concat_axis=1)
h9_conv_V = BatchNormalization(axis=1)(h9_conv_V)

h10_conv_Y_a = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_Y)
h10_conv_Y_b = Convolution2D(40, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_Y)
h10_conv_Y = merge([h10_conv_Y_a, h10_conv_Y_b], mode='max', concat_axis=1)
h10_conv_Y = BatchNormalization(axis=1)(h10_conv_Y)

h10_conv_U_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_U)
h10_conv_U_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_U)
h10_conv_U = merge([h10_conv_U_a, h10_conv_U_b], mode='max', concat_axis=1)
h10_conv_U = BatchNormalization(axis=1)(h10_conv_U)

h10_conv_V_a = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_V)
h10_conv_V_b = Convolution2D(20, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h9_conv_V)
h10_conv_V = merge([h10_conv_V_a, h10_conv_V_b], mode='max', concat_axis=1)
h10_conv_V = BatchNormalization(axis=1)(h10_conv_V)

poolY = MaxPooling2D(pool_size=(2, 2))(h10_conv_Y)
poolU = MaxPooling2D(pool_size=(2, 2))(h10_conv_U)
poolV = MaxPooling2D(pool_size=(2, 2))(h10_conv_V)

poolY = Dropout(0.2)(poolY)
poolU = Dropout(0.2)(poolU)
poolV = Dropout(0.2)(poolV)

# Cross connections: Y <-> U, Y <-> V
Y_to_UV_a = Convolution2D(40, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
Y_to_UV_b = Convolution2D(40, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolY)
Y_to_UV = merge([Y_to_UV_a, Y_to_UV_b], mode='max', concat_axis=1)
Y_to_UV = BatchNormalization(axis=1)(Y_to_UV)

U_to_Y_a = Convolution2D(20, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
U_to_Y_b = Convolution2D(20, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolU)
U_to_Y = merge([U_to_Y_a, U_to_Y_b], mode='max', concat_axis=1)
U_to_Y = BatchNormalization(axis=1)(U_to_Y)

V_to_Y_a = Convolution2D(20, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
V_to_Y_b = Convolution2D(20, 1, 1, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(poolV)
V_to_Y = merge([V_to_Y_a, V_to_Y_b], mode='max', concat_axis=1)
V_to_Y = BatchNormalization(axis=1)(V_to_Y)

Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=1)
Umap = merge([poolU, Y_to_UV], mode='concat', concat_axis=1)
Vmap = merge([poolV, Y_to_UV], mode='concat', concat_axis=1)

h11_conv_Y_a = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
h11_conv_Y_b = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Ymap)
h11_conv_Y = merge([h11_conv_Y_a, h11_conv_Y_b], mode='max', concat_axis=1)
h11_conv_Y = BatchNormalization(axis=1)(h11_conv_Y)

h11_conv_U_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
h11_conv_U_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Umap)
h11_conv_U = merge([h11_conv_U_a, h11_conv_U_b], mode='max', concat_axis=1)
h11_conv_U = BatchNormalization(axis=1)(h11_conv_U)

h11_conv_V_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
h11_conv_V_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(Vmap)
h11_conv_V = merge([h11_conv_V_a, h11_conv_V_b], mode='max', concat_axis=1)
h11_conv_V = BatchNormalization(axis=1)(h11_conv_V)

h12_conv_Y_a = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_Y)
h12_conv_Y_b = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_Y)
h12_conv_Y = merge([h12_conv_Y_a, h12_conv_Y_b], mode='max', concat_axis=1)
h12_conv_Y = BatchNormalization(axis=1)(h12_conv_Y)

h12_conv_U_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_U)
h12_conv_U_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_U)
h12_conv_U = merge([h12_conv_U_a, h12_conv_U_b], mode='max', concat_axis=1)
h12_conv_U = BatchNormalization(axis=1)(h12_conv_U)

h12_conv_V_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_V)
h12_conv_V_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h11_conv_V)
h12_conv_V = merge([h12_conv_V_a, h12_conv_V_b], mode='max', concat_axis=1)
h12_conv_V = BatchNormalization(axis=1)(h12_conv_V)

h13_conv_Y_a = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_Y)
h13_conv_Y_b = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_Y)
h13_conv_Y = merge([h13_conv_Y_a, h13_conv_Y_b], mode='max', concat_axis=1)
h13_conv_Y = BatchNormalization(axis=1)(h13_conv_Y)

h13_conv_U_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_U)
h13_conv_U_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_U)
h13_conv_U = merge([h13_conv_U_a, h13_conv_U_b], mode='max', concat_axis=1)
h13_conv_U = BatchNormalization(axis=1)(h13_conv_U)

h13_conv_V_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_V)
h13_conv_V_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h12_conv_V)
h13_conv_V = merge([h13_conv_V_a, h13_conv_V_b], mode='max', concat_axis=1)
h13_conv_V = BatchNormalization(axis=1)(h13_conv_V)

h14_conv_Y_a = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_Y)
h14_conv_Y_b = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_Y)
h14_conv_Y = merge([h14_conv_Y_a, h14_conv_Y_b], mode='max', concat_axis=1)
h14_conv_Y = BatchNormalization(axis=1)(h14_conv_Y)

h14_conv_U_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_U)
h14_conv_U_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_U)
h14_conv_U = merge([h14_conv_U_a, h14_conv_U_b], mode='max', concat_axis=1)
h14_conv_U = BatchNormalization(axis=1)(h14_conv_U)

h14_conv_V_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_V)
h14_conv_V_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h13_conv_V)
h14_conv_V = merge([h14_conv_V_a, h14_conv_V_b], mode='max', concat_axis=1)
h14_conv_V = BatchNormalization(axis=1)(h14_conv_V)

h15_conv_Y_a = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_Y)
h15_conv_Y_b = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_Y)
h15_conv_Y = merge([h15_conv_Y_a, h15_conv_Y_b], mode='max', concat_axis=1)
h15_conv_Y = BatchNormalization(axis=1)(h15_conv_Y)

h15_conv_U_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_U)
h15_conv_U_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_U)
h15_conv_U = merge([h15_conv_U_a, h15_conv_U_b], mode='max', concat_axis=1)
h15_conv_U = BatchNormalization(axis=1)(h15_conv_U)

h15_conv_V_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_V)
h15_conv_V_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h14_conv_V)
h15_conv_V = merge([h16_conv_V_a, h15_conv_V_b], mode='max', concat_axis=1)
h15_conv_V = BatchNormalization(axis=1)(h15_conv_V)

h16_conv_Y_a = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_Y)
h16_conv_Y_b = Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_Y)
h16_conv_Y = merge([h16_conv_Y_a, h16_conv_Y_b], mode='max', concat_axis=1)
h16_conv_Y = BatchNormalization(axis=1)(h16_conv_Y)

h16_conv_U_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_U)
h16_conv_U_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_U)
h16_conv_U = merge([h16_conv_U_a, h16_conv_U_b], mode='max', concat_axis=1)
h16_conv_U = BatchNormalization(axis=1)(h16_conv_U)

h16_conv_V_a = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_V)
h16_conv_V_b = Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform', W_regularizer=l2(0.0005))(h15_conv_V)
h16_conv_V = merge([h16_conv_V_a, h16_conv_V_b], mode='max', concat_axis=1)
h16_conv_V = BatchNormalization(axis=1)(h16_conv_V)

poolY = MaxPooling2D(pool_size=(8, 8))(h16_conv_Y)
poolU = MaxPooling2D(pool_size=(8, 8))(h16_conv_U)
poolV = MaxPooling2D(pool_size=(8, 8))(h16_conv_V)

poolY = Dropout(0.2)(poolY)
poolU = Dropout(0.2)(poolU)
poolV = Dropout(0.2)(poolV)

concat_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=1)

h16 = Flatten()(concat_map)
h17 = MaxoutDense(500, nb_feature=5, init='glorot_uniform', W_regularizer=l2(0.0005))(h16)
h17 = BatchNormalization(axis=1)(h17)
h17_drop = Dropout(0.2)(h17)
out = Dense(10, activation='softmax', init='glorot_uniform', W_regularizer=l2(0.0005))(h17)

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

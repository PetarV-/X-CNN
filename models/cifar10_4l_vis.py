from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, merge, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
#from keras.utils.visualize_util import plot
from utils.preprocess_vis import get_cifar
from utils.palette import get_palette
import numpy as np
import scipy.misc
import PIL.Image as im



batch_size = 32
nb_classes = 10
nb_epoch = 2
train_ratio=0.2
data_augmentation = False

# plot the model?
plot_model = True
show_shapes = True
plot_file = 'cifar10_4l.png'
depthY=32
depthUV=16
# show the summary?
show_summary = True

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test), X_train_rgb = get_cifar(p=train_ratio, append_test=False, use_c10=True)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(Y_train.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


#cross-connections between two conv layers, Y is the middle layer, while U and V are side layers.

inputYUV = Input(shape=(3, 32, 32))
inputNorm = BatchNormalization(axis=1)(inputYUV)

# To simplify the data augmentation, I delay slicing until this point.
# Not sure if there is a better way to handle it. ---Petar
inputY = Lambda(lambda x: x[:,0:1,:,:], output_shape=(1, 32, 32))(inputNorm)
inputU = Lambda(lambda x: x[:,1:2,:,:], output_shape=(1, 32, 32))(inputNorm)
inputV = Lambda(lambda x: x[:,2:3,:,:], output_shape=(1, 32, 32))(inputNorm)

convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

poolY = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(convY)
poolU = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(convU)
poolV = MaxPooling2D((2,2), strides=(2, 2), border_mode='same')(convV)

poolY_1 = Dropout(0.25)(poolY)
poolU_1 = Dropout(0.25)(poolU)
poolV_1 = Dropout(0.25)(poolV)

U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU_1)
V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV_1)
Y_to_UV = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY_1)

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
model.load_weights('cifar10_4l.h5')

#model for extracting intermediate feaature maps
model_Y_pre = Model(input=inputYUV, output=poolY_1)
model_Y_post = Model(input=inputYUV, output=Y_to_UV)
model_U_pre = Model(input=inputYUV, output=poolU_1)
model_U_post = Model(input=inputYUV, output=U_to_Y)
model_V_pre = Model(input=inputYUV, output=poolV_1)
model_V_post = Model(input=inputYUV, output=V_to_Y)

#draw a random image id
from random import randint
img_to_visualize = randint(0, len(X_train) - 1)

#get feature maps for yuv channels before and after cross connections
fmap_Y_pre= model_Y_pre.predict(X_train[img_to_visualize:img_to_visualize+1])
fmap_Y_post=model_Y_post.predict(X_train[img_to_visualize:img_to_visualize+1])
fmap_U_pre= model_U_pre.predict(X_train[img_to_visualize:img_to_visualize+1])
fmap_U_post=model_U_post.predict(X_train[img_to_visualize:img_to_visualize+1])
fmap_V_pre= model_V_pre.predict(X_train[img_to_visualize:img_to_visualize+1])
fmap_V_post=model_V_post.predict(X_train[img_to_visualize:img_to_visualize+1])

print("Image used: #%d (label=%d)" % (img_to_visualize, np.argmax(Y_train[img_to_visualize])))
print(np.max(fmap_Y_pre),np.min(fmap_Y_post))
#save original image and yuv channels
Ychannel=X_train[img_to_visualize,0,:,:]
Ychannel=Ychannel+np.abs(np.min(Ychannel))
scipy.misc.imsave('imageY.jpg',X_train[img_to_visualize,0,:,:])
scipy.misc.imsave('imageU.jpg',X_train[img_to_visualize,1,:,:])
scipy.misc.imsave('imageV.jpg',X_train[img_to_visualize,2,:,:])
scipy.misc.imsave('image.jpg',np.flipud(np.rot90(np.transpose(X_train_rgb[img_to_visualize]))))

#get palette for colored combination of feature map layers
palette_Y=get_palette(depthY)
palette_UV=get_palette(depthUV)

#initialize all combined feature maps
fmap_Y_pre_combined=np.zeros((16,16,3))
fmap_Y_post_combined=np.zeros((16,16,3))
fmap_U_pre_combined=np.zeros((16,16,3))
fmap_U_post_combined=np.zeros((16,16,3))
fmap_V_pre_combined=np.zeros((16,16,3))
fmap_V_post_combined=np.zeros((16,16,3))

#combine for Y channel
for i in range(depthY):
    fmap_pre_slice=fmap_Y_pre[0][i]
    fmap_pre_slice_color=np.repeat(np.expand_dims(fmap_pre_slice,axis=2),3,axis=2)*palette_Y[i]
    fmap_Y_pre_combined+=fmap_pre_slice_color/depthY
    fmap_post_slice=fmap_Y_post[0][i]
    fmap_post_slice_color=np.repeat(np.expand_dims(fmap_post_slice,axis=2),3,axis=2)*palette_Y[i]
    fmap_Y_post_combined+=fmap_post_slice_color/depthY

for layer in model.layers:
    if layer.name in ['convolution2d_7','convolution2d_8','convolution2d_9']:
        weight_csvname=layer.name+'.csv'
        weights=np.zeros((len(layer.get_weights()[0]),len(layer.get_weights()[0][0])))
	#print(len(layer.get_weights()[0][0][0]))
        for i in range(len(layer.get_weights()[0])):
	    #print(layer.get_weights()[i])
            for j in range(len(layer.get_weights()[0][0])):
                weights[i,j]=layer.get_weights()[0][i][j][0]
	#reshaped=np.reshape(layer.get_weights(),layer.get_weights().shape[0:2])
        np.savetxt(weight_csvname,weights,delimiter=',')

#print(np.min(fmap_Y_pre_combined),np.min(fmap_Y_post_combined))
difmap_Y_pos=np.clip(fmap_Y_post_combined-fmap_Y_pre_combined,0,1)
difmap_Y_neg=np.clip(fmap_Y_pre_combined-fmap_Y_post_combined,0,1)
#combine for U and V channel
for i in range(depthUV):
    fmap_pre_slice=fmap_U_pre[0][i]
    fmap_pre_slice_color=np.repeat(np.expand_dims(fmap_pre_slice,axis=2),3,axis=2)*palette_UV[i]
    fmap_U_pre_combined+=fmap_pre_slice_color/depthY
    fmap_post_slice=fmap_U_post[0][i]
    fmap_post_slice_color=np.repeat(np.expand_dims(fmap_post_slice,axis=2),3,axis=2)*palette_UV[i]
    fmap_U_post_combined+=fmap_post_slice_color/depthY

    fmap_pre_slice=fmap_V_pre[0][i]
    fmap_pre_slice_color=np.repeat(np.expand_dims(fmap_pre_slice,axis=2),3,axis=2)*palette_UV[i]
    fmap_V_pre_combined+=fmap_pre_slice_color/depthY
    fmap_post_slice=fmap_V_post[0][i]
    fmap_post_slice_color=np.repeat(np.expand_dims(fmap_post_slice,axis=2),3,axis=2)*palette_UV[i]
    fmap_V_post_combined+=fmap_post_slice_color/depthY

difmap_U_pos=np.clip(fmap_U_post_combined-fmap_U_pre_combined,0,1)
difmap_U_neg=np.clip(fmap_U_pre_combined-fmap_U_post_combined,0,1)
difmap_V_pos=np.clip(fmap_V_post_combined-fmap_V_pre_combined,0,1)
difmap_V_neg=np.clip(fmap_V_pre_combined-fmap_V_post_combined,0,1)
#save image to files
scipy.misc.imsave('fmap_Y_pre_combined.jpg',fmap_Y_pre_combined)
scipy.misc.imsave('fmap_Y_post_combined.jpg',fmap_Y_post_combined)
scipy.misc.imsave('difmap_Y_pos.jpg',difmap_Y_pos)
scipy.misc.imsave('difmap_Y_neg.jpg',difmap_Y_neg)
scipy.misc.imsave('fmap_U_pre_combined.jpg',fmap_Y_pre_combined)
scipy.misc.imsave('fmap_U_post_combined.jpg',fmap_Y_post_combined)
scipy.misc.imsave('difmap_U_pos.jpg',difmap_U_pos)
scipy.misc.imsave('difmap_U_neg.jpg',difmap_U_neg)
scipy.misc.imsave('fmap_V_pre_combined.jpg',fmap_Y_pre_combined)
scipy.misc.imsave('fmap_V_post_combined.jpg',fmap_Y_post_combined)
scipy.misc.imsave('difmap_V_pos.jpg',difmap_V_pos)
scipy.misc.imsave('difmap_V_neg.jpg',difmap_V_neg)
#plt.show()





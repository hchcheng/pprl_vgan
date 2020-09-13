import os, random
import dlib

from keras import backend as K
#os.environ["KERAS_BACKEND"] = "tensorflow"

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("tensorflow")

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import h5py
import numpy as np
from keras.utils import plot_model

from keras.layers import Input, merge, Lambda
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D, \
    Conv2DTranspose
from keras.layers.normalization import *
from keras.optimizers import *
from keras import initializers
import matplotlib.pyplot as plt
import pickle, random, sys, keras
from keras.models import Model
from tqdm import tqdm
import time
import os, sys
from functools import partial

import random

import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system names
        if name == "PIL":
            name = "Pillow"
        elif name == "sklearn":
            name = "scikit-learn"

        yield name
imports = list(set(get_imports()))

requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))
    
print('keras=='+keras.__version__)


#############################################################

# normal = partial(initializers.normal, scale=.02)

##### load and preprocess the dataset ##
batch_size = 256
num_ep = 7 # number of facial expressions (referenced in y_train2)
num_pp = 6 # number of identities (referenced in y_train1)
epochs = 300
img_rows, img_cols = 64, 64
c_dim = num_pp
date = 2020

# Extracting data from processed h5py file
print ('Loading data...')
f = h5py.File('../input/processed-dataset-512-128/processed_dataset.h5','r')
print ('Finished loading...')

dataset_size = 26880
training_size = 21504
test_size = 5376

# Extracting information from htf5 files
x_train = f['x_train'][()][:]
x_test  = f['x_test'][()][:]
y_train1 = f['y_train1'][()][:]
y_test1  = f['y_test1'][()][:]
y_train2 = f['y_train2'][()][:]
y_test2  = f['y_test2'][()][:]

# Change data into correct form
y_train1 = keras.utils.to_categorical(y_train1, num_pp)
y_test1  = keras.utils.to_categorical(y_test1, num_pp)
y_train2 = keras.utils.to_categorical(y_train2, num_ep)
y_test2  = keras.utils.to_categorical(y_test2, num_ep)

x_ori = np.divide(x_train- 127.5, 127.5)
###########################################################

epsilon_std = 1
def sampling_np(args):
    z_mean, z_log_var = args
    epsilon = np.random.normal(loc=0., scale=epsilon_std, size=(z_mean.shape[0], z_dim), )
    return z_mean + np.exp(z_log_var / 2) * epsilon


def generate_dataset(ee):
    ## save to numpyz##############
    c = np.random.randint(num_pp, size=x_ori.shape[0])
    c_train = keras.utils.to_categorical(c, num_pp)
    c = np.random.randint(num_pp, size=x_test.shape[0])
    c_test = keras.utils.to_categorical(c, num_pp)

    [z_train, mean_var_train, h1, h2, h3, h4] = encoder.predict(x_ori)
    encoded_xtrain = decoder.predict([z_train, c_train, h1, h2, h3, h4])

    [z_test, mean_var_test, h1, h2, h3, h4] = encoder.predict(x_test)
    encoded_xtest = decoder.predict([z_test, c_test, h1, h2, h3, h4])

    np.savez('/Z_' + str(date) + 'epoch'+str(ee)+'_64_64_VAE_GAN_labelfull_v2.npz',
             encoded_xtrain, y_train1, y_train2, c_train, encoded_xtest, y_test1, y_test2, c_test)
    np.savez('/X_' + str(date) + 'epoch'+str(ee)+ '_fi_512_VAE_GAN_labelfull_v2.npz',
             z_train, y_train1, y_train2, c_train, z_test, y_test1, y_test2, c_test)

opt  = RMSprop(lr=0.0003, decay=1e-6)
dopt = RMSprop(lr=0.0003, decay=1e-6)


def KL_loss(y_true, y_pred):
    z_mean = y_pred[:, 0:z_dim]
    z_log_var = y_pred[:, z_dim:2 * z_dim]
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], z_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(K.square(z_log_sigma) / 2) * epsilon


def model_encoder(z_dim, input_shape, units=512, dropout=0.3):
    k = 8
    x = Input(input_shape)
    print(x.shape)
    h = Conv2D(units / 8, (k, k), strides=(2, 2), padding='same')(x)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    # h = MaxPooling2D(pool_size=(2, 2))(h)
    h1 = LeakyReLU(0.2)(h)
    print(h1.shape)
    h2 = Conv2D(units / 4, (k, k), strides=(2, 2), padding='same')(h1)
    h2 = BatchNormalization(momentum=0.8)(h2)
    h2 = Dropout(dropout)(h2)
    # h = MaxPooling2D(pool_size=(2, 2))(h)
    h2 = LeakyReLU(0.2)(h2)
    print(h2.shape)
    h3 = Conv2D(units / 2, (k, k), strides=(2, 2), padding='same')(h2)
    h3 = BatchNormalization(momentum=0.8)(h3)
    h3 = Dropout(dropout)(h3)
    # h = MaxPooling2D(pool_size=(2, 2))(h)
    h3 = LeakyReLU(0.2)(h3)
    print(h3.shape)
    h4 = Conv2D(units, (k, k), strides=(2, 2), padding='same')(h3)
    h4 = BatchNormalization(momentum=0.8)(h4)
    h4 = Dropout(dropout)(h4)
    h4 = LeakyReLU(0.2)(h4)
    print(h4.shape)
    h5 = Flatten()(h4)
    mean = Dense(z_dim, name="encoder_mean")(h5)
    logvar = Dense(z_dim, name="encoder_sigma", activation='sigmoid')(h5)

    z = Lambda(sampling, output_shape=(z_dim,))([mean, logvar])
    h6 = keras.layers.concatenate([mean, logvar])
    return Model(x, [z, h6, h1, h2, h3, h4], name='Encoder')

def model_decoder(z_dim, c_dim):
    k = 8
    x = Input(shape=(z_dim,))
    auxiliary_c = Input(shape=(c_dim,), name='aux_input_c')
    h1= Input(shape=(32,32,32), name='skip1')
    h2= Input(shape=(16,16,64), name='skip2')
    h3= Input(shape=(8,8,128), name='skip3')
    h4= Input(shape=(4,4,256), name='skip4')
    h = keras.layers.concatenate([x, auxiliary_c])
    h = Dense(4 * 4 * 128, activation='relu')(h)
    h = Reshape((4, 4, 128))(h)
    # h = LeakyReLU(0.2)(h)
    h = keras.layers.concatenate([h, h4])
    h = Conv2DTranspose(units, (k, k), strides=(2, 2), padding='same', activation='relu')(h)  # 32*32*64
    # h = Dropout(dropout)(h)
    h = BatchNormalization(momentum=0.8)(h)
    # h = LeakyReLU(0.2)(h)
    # h = UpSampling2D(size=(2, 2))(h)
    #print(h.shape)
    h = keras.layers.concatenate([h, h3])
    h = Conv2DTranspose(units / 2, (k, k), strides=(2, 2), padding='same', activation='relu')(h)  # 64*64*64
    # h = Dropout(dropout)(h)
    h = BatchNormalization(momentum=0.8)(h)
    # h = LeakyReLU(0.2)(h)
    # h = UpSampling2D(size=(2, 2))(h)
    print(h.shape)
    h = keras.layers.concatenate([h, h2])
    h = Conv2DTranspose(units / 2, (k, k), strides=(2, 2), padding='same', activation='relu')(h)  # 8*6*64
    # h = Dropout(dropout)(h)
    h = BatchNormalization(momentum=0.8)(h)

    print(h.shape)
    h = keras.layers.concatenate([h, h1])
    h = Conv2DTranspose(3, (k, k), strides=(2, 2), padding='same', activation='tanh')(h)  # 8*6*64
    return Model([x, auxiliary_c, h1, h2, h3, h4], h, name="Decoder")





################################################ Build the discrminator ###########################################################################

input_shape = (img_rows, img_cols, 3)
loss_weights_1= Input(shape=(1,), name='disc_1')
loss_weights_2= Input(shape=(1,),name='disc_2')
loss_weights_3= Input(shape=(1,),name='disc_3')
targets1  = Input(shape = (1,),name='disc_4')
targets2  = Input(shape = (num_pp,),name='disc_5')
targets3  = Input(shape = (num_ep,),name='disc_6')
d_input   = Input(input_shape,name='disc_7')
rep_field = 8
x = Conv2D(32, (rep_field, rep_field), strides=(2, 2), padding='same', name='id_conv1')(d_input)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)
# x  = AveragePooling2D((2, 2), padding='same')(x)
# x  = Dropout(0.3)(x)

x = Conv2D(64, (rep_field, rep_field), strides=(2, 2), padding='same', name='id_conv2')(x)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)
# x  = AveragePooling2D((2, 2), padding='same')(x)
# x  = Dropout(0.3)(x)

x = Conv2D(128, (rep_field, rep_field), strides=(2, 2), padding='same', name='id_conv3')(x)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)
# x  = AveragePooling2D((2, 2), padding='same')(x)
# x  = Dropout(0.3)(x)

x = Conv2D(256, (rep_field, rep_field), strides=(2, 2), padding='same', name='id_conv4')(x)
# x = BatchNormalization(momentum=0.9)(x)
x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(256, name='ds')(x)
x = LeakyReLU(0.2)(x)
x = Dropout(0.5)(x)
output_binary     = Dense(1, activation='sigmoid', name='bin_real')(x)
output_identity   = Dense(num_pp, activation='softmax', name='id_real')(x)
output_expression = Dense(num_ep, activation='softmax', name='exp_real')(x)

discriminator = Model([d_input, loss_weights_1, loss_weights_2,loss_weights_3, targets1, targets2, targets3], [output_binary, output_identity, output_expression])

from keras import losses

loss =loss_weights_1*losses.binary_crossentropy(targets1,output_binary) + \
      loss_weights_2*losses.categorical_crossentropy(targets2,output_identity)+ \
      loss_weights_3*losses.categorical_crossentropy(targets3,output_expression)
discriminator.add_loss(loss)
#discriminator.load_weights('/content/drive/My Drive/pprl_vgan/trained_weights_01/discriminator_2020epochs2000.h5')
discriminator.compile( optimizer=dopt, loss = None)
# discriminator.summary()
print (discriminator.metrics_names)
#plot_model(discriminator, to_file = '/media/vivo/New Volume/FERG_DB_256/stats/disc_0605_model.png')

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


make_trainable(discriminator, False)
discriminator.trainable = False

# #### Build GAN model ####
z_dim = 128
units = 256
GANloss_weights_vae = Input(shape = (1,))
GANtargets_vae  = Input(shape = (z_dim*2,))

#ee = 100

auxiliary_c = Input(shape=(c_dim,), name='aux_input_c')
encoder = model_encoder(z_dim=z_dim, input_shape=(img_rows, img_cols, 3), units=units, dropout=0.3)
#encoder.load_weights('/content/drive/My Drive/pprl_vgan/trained_weights_01/encoder_2020epochs2000.h5')
encoder.compile(loss='binary_crossentropy', optimizer=opt)
# encoder.summary()

decoder = model_decoder(z_dim=z_dim, c_dim=c_dim)
#decoder.load_weights('/content/drive/My Drive/pprl_vgan/trained_weights_01/decoder_2020epochs2000.h5')
decoder.compile(loss='binary_crossentropy', optimizer=opt)
# decoder.summary()

# The VAE network takes the output image and feeds it through the generator network one more time.
# The final output after this is compared with the original image, and the loss is backpropagated through the network
# which incentivizes weights which can transfer facial expressions between faces.
# The initial output image is x, and the initial input is ori_x which is compared with the output of vae

temp_x = Input(input_shape)
temp_ori_x = Input(input_shape)
temp_c = Input(shape=(c_dim,), name='temp_c')
vae_output = decoder([encoder(temp_x)[0], temp_c, encoder(temp_x)[2], encoder(temp_x)[3], encoder(temp_x)[4], encoder(temp_x)[5]])

vae = Model([temp_x, temp_c, temp_ori_x], vae_output, name='VAE')
vae_loss =losses.binary_crossentropy(temp_ori_x, vae_output) #compare original image to final image
vae.add_loss(vae_loss)
vae.compile(loss=None, optimizer=opt)
# vae.summary()

### Generate Image set ###
# generate_dataset(ee=ee)
###


### GAN formulation ###
[z, mean_var, h1, h2, h3, h4] = encoder(d_input)
xpred = decoder([z, auxiliary_c, h1, h2, h3, h4])
output_binary, output_identity, output_expression = discriminator([xpred, loss_weights_1, loss_weights_2,loss_weights_3, targets1, targets2, targets3])
GAN = Model([d_input, auxiliary_c, GANloss_weights_vae, loss_weights_1,loss_weights_2,loss_weights_3, GANtargets_vae, targets1, targets2, targets3],\
            [mean_var, output_binary, output_identity, output_expression])

GANloss = GANloss_weights_vae*KL_loss(GANtargets_vae, mean_var) + \
          loss_weights_1*losses.binary_crossentropy(targets1,output_binary) + \
          loss_weights_2*losses.categorical_crossentropy(targets2, output_identity)+ \
          loss_weights_3*losses.categorical_crossentropy(targets3, output_expression)
GAN.add_loss(GANloss)
#GAN.load_weights('/content/drive/My Drive/pprl_vgan/trained_weights_01/VAEGAN_2020epochs2000.h5')
GAN.compile(optimizer = opt, loss = None)
#GAN = tf.keras.models.load_model('/content/drive/My Drive/pprl_vgan/trained_weights_01/VAEGAN_2020epochs2000.h5')
# GAN.summary()
print (GAN.metrics_names)


# plot_model(GAN, to_file = 'GAN_model.png')

# def plotGeneratedImages(epoch, idx=0, examples=10, dim=(10, 10), figsize=(10, 10)):
#     n = num_pp  # how many digits we will display
#     pp_avg = 4500
#     plt.figure(figsize=(16, 4))
#
#
#     sample = x_ori[idx:idx + n, :, :, :]
#     c = np.asarray([0, 1, 2, 3, 4, 5])
#     c = keras.utils.to_categorical(c, num_pp)
#
#     [z, mean_var] = encoder.predict(sample)
#     generated_images = decoder.predict([z, c])
#
#     for i in range(n):
#         # display original
#         ax = plt.subplot(2, n, i + 1)
#         ori = sample[i].reshape(img_rows, img_cols, 3)
#         ori = np.uint8(ori * 127.5 + 127.5)
#         plt.imshow(ori)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#         # display reconstruction
#         ax = plt.subplot(2, n, i + 1 + n)
#         rec = generated_images[i].reshape(img_rows, img_cols, 3)
#         rec = np.uint8(rec * 127.5 + 127.5)
#         plt.imshow(rec)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#     # Path to be created
#     plt.savefig(path + '/GAN_MUG_results_' + str(date) + '_generated_image_epoch_%d.png' % epoch)
#     plt.close()
def plotGeneratedImages(epoch, idx=0, examples=10, dim=(10, 10), figsize=(10, 10)):
    n = num_pp * 2  # how many digits we will display
    pp_avg = 4500
    plt.figure(figsize=(16, 4))

    ####I added
    idx = [260, 50, 641, 260, 1120, 5471, 9125, 14012, 12547, 18425, 21000, 8236]
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # detect faces in the grayscale image
    import cv2
    #from imutils import face_utils
    # img = x_ori[10100, :, :, :].copy()
    # gray = (cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)*255).astype(np.uint8)
    # rect = detector(gray, 1)
    # shape = predictor(gray, rect[0])
    # shape = face_utils.shape_to_np(shape)
    # for (x, y) in shape:
    #     img[y,x] = (0, 0, 255)
    # plt.imshow(img)


    sample = x_ori[idx, :, :, :]
    # sample = np.divide(x_test, 255)[idx, :, :, :]
    # sample = x_ori[idx:idx + n, :, :, :]
    c = np.asarray([0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0])  # output identities
    c = keras.utils.to_categorical(c, num_pp)

    [z, mean_var, h1, h2, h3, h4] = encoder.predict(sample)
    generated_images = decoder.predict([z, c, h1, h2, h3, h4])

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        ori = sample[i].reshape(img_rows, img_cols, 3)
        ori = np.uint8(ori * 127.5 + 127.5)
        plt.imshow(ori)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        rec = generated_images[i].reshape(img_rows, img_cols, 3)
        rec = np.uint8(rec * 127.5 + 127.5)
        plt.imshow(rec)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Path to be created
    # plt.savefig(path + '/GAN_MUG_results_' + str(date) + '_generated_image_epoch_%d.tif' % epoch)
    # plt.savefig()
    # plt.close()

    plt.savefig(path + '/GAN_MUG_results_' + str(date) + '_generated_image_epoch_%d.png' % epoch)
    plt.close()
    # f = h5py.File('GAN_MUG_results_' + str(date) + '_generated_image_epoch_%d.tif' % epoch, 'w')

# Check discriminator accuracy
def discriminator_acc():
    X_test = np.divide(x_test-127.5, 127.5)
    size = len(X_test)

    binary_acc = 0
    identity_acc = 0
    expression_acc = 0

    batchCount = int(X_test.shape[0] / batch_size)
    for i in range(batchCount):
        # idx = random.sample(range(0, x_test.shape[0]), size)
        idx = np.arange(i*batch_size,(i+1)*batch_size)
        image_batch = X_test[idx, :, :, :]

        c = np.random.randint(num_pp, size=batch_size)
        c = keras.utils.to_categorical(c, num_pp)


        [z, mean_var, h1, h2, h3, h4] = encoder.predict(image_batch)
        generated_images = decoder.predict([z, c, h1, h2, h3, h4])

        y1_batch = c#y_train1[idx, :]
        y2_batch = y_test2[idx, :]
        y0_batch = np.zeros((size, 1))

        loss_weights_1 = np.ones(shape=(size,)) * 1 / 4.0
        loss_weights_2 = np.ones(shape=(size,)) * 1 / 2.0
        loss_weights_3 = np.ones(shape=(size,)) * 1 / 4.0

        [out_binary, out_identity, out_expression] = discriminator.predict(
            [generated_images, loss_weights_1, loss_weights_2, loss_weights_3, y0_batch, y1_batch, y2_batch])

        out_binary = np.round(out_binary, 0)
        # out_identity=np.round(out_identity,0)
        # out_expression=np.round(out_expression,0)

        for i in range(batch_size):
            if (y0_batch[i] != out_binary[i]):
                binary_acc += 1

            ii = np.where(out_identity[i] == max(out_identity[i]))
            # print(ii)
            for j in range(len(out_identity[i])):
                out_identity[i, j] = 0
            out_identity[i, ii] = 1
            comp = y1_batch[i] == out_identity[i]
            if (not comp.all()):
                identity_acc += 1

            ii = np.where(out_expression[i] == max(out_expression[i]))
            # print(ii)
            for j in range(len(out_expression[i])):
                out_expression[i, j] = 0
            out_expression[i, ii] = 1
            comp = y2_batch[i] == out_expression[i]
            if (not comp.all()):
                expression_acc += 1

            # print(y2_batch[i])
            # print(out_expression[i])

    binary_acc = 1 - (binary_acc / size)
    identity_acc = 1 - (identity_acc / size)
    expression_acc = 1 - (expression_acc / size)

    return ([binary_acc, identity_acc, expression_acc])


def train_for_n(nb_epoch=50000, plt_frq=25, BATCH_SIZE=256):
    batchCount = int(x_ori.shape[0] / BATCH_SIZE)
    for ee in range(1, nb_epoch + 1):
        print('-' * 15, 'Epoch %d' % ee, '-' * 15)
        plotGeneratedImages(epoch=ee, idx=75)
        # val_bin_acc, val_id_acc, val_ep_acc = val_test()
        for e in tqdm(range(batchCount)):
            # for didx in xrange(0,k):
            idx = random.sample(range(0, x_ori.shape[0]),
                                BATCH_SIZE)  # train discriminator twice more than the generator
            image_batch = x_ori[idx, :, :, :]  # real data
            c = np.random.randint(num_pp, size=BATCH_SIZE)
            c = keras.utils.to_categorical(c, num_pp)

            [z, mean_var, h1, h2, h3, h4] = encoder.predict(image_batch)
            generated_images = decoder.predict([z, c, h1, h2, h3, h4])

            y1_batch = y_train1[idx, :]
            y2_batch = y_train2[idx, :]

            # generated_images = generator.predict([image_batch, c_, z])
            y0_dist_real = np.random.uniform(0.9, 1.0, size=[BATCH_SIZE, 1])
            y0_dist_fake = np.random.uniform(0, 0.1, size=[BATCH_SIZE, 1])



            make_trainable(discriminator, True)
            discriminator.trainable = True
            loss_weights_1 = np.ones(shape = (batch_size,))*1/4.0
            loss_weights_2 = np.ones(shape = (batch_size,))*1/2.0
            loss_weights_3 = np.ones(shape = (batch_size,))*1/4.0
            d_loss_real = discriminator.train_on_batch([image_batch, loss_weights_1, loss_weights_2, loss_weights_3,y0_dist_real, y1_batch, y2_batch],y= None)
            loss_weights_1 = np.ones(shape=(batch_size,))*1.0
            loss_weights_2 = np.ones(shape=(batch_size,)) * 0
            loss_weights_3 = np.ones(shape=(batch_size,)) * 0
            d_loss_fake = discriminator.train_on_batch([generated_images,loss_weights_1,loss_weights_2,loss_weights_3, y0_dist_fake, c, y2_batch], y = None)


            make_trainable(discriminator, False)
            discriminator.trainable = False
            for ii in range(0, 3):
                idx = random.sample(range(0, x_ori.shape[0]),
                                    BATCH_SIZE)  # train discriminator twice more than the generator
                image_batch = x_ori[idx, :, :, :]  # real data
                c = np.random.randint(num_pp, size=BATCH_SIZE)
                c = keras.utils.to_categorical(c, num_pp)

                mean_var_ref = np.ones((BATCH_SIZE, z_dim * 2))
                y1_batch = y_train1[idx, :]
                y2_batch = y_train2[idx, :]

                y0_batch = np.ones((BATCH_SIZE, 1)) #0.002, 0.09, 0.8, 0.108
                GANloss_weights_vae = np.ones(shape = (batch_size,))*0.002
                loss_weights_1 = np.ones(shape = (batch_size,))*0.078
                loss_weights_2 = np.ones(shape = (batch_size,))*0.8
                loss_weights_3 = np.ones(shape = (batch_size,))*0.12
                g_loss = GAN.train_on_batch([image_batch, c, GANloss_weights_vae, loss_weights_1, loss_weights_2, loss_weights_3, mean_var_ref, y0_batch, c, y2_batch], y = None)
                vae_loss = vae.train_on_batch([generated_images, y1_batch, image_batch]) # output images, initial identities, initial images


        if ee % 25 == 0:
            GAN.save(path +'/VAEGAN_real_' + str(date) + 'epochs' + str(
                ee) + '.h5')
            encoder.save(path +'/encoder_MUG_8pp_real_' + str(
                date) + 'epochs' + str(ee) + '.h5')
            decoder.save(path +'/decoder_MUG_8pp_real_' + str(
                date) + 'epochs' + str(ee) + '.h5')
            discriminator.save(path +'/discriminator_MUG_8pp_real_' + str(
                date) + 'epochs' + str(ee) + '.h5')
        if ee % 10 == 0:
            print(discriminator_acc())



start_time = time.time()
path = "./path_" + str(date)
if os.path.isdir(path) == False:
    os.mkdir(path);

print(discriminator_acc())

train_for_n(nb_epoch=200, plt_frq=500, BATCH_SIZE=batch_size)

process_time = time.time() - start_time
print("Elapsed: %s " % (process_time))
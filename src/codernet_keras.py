from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, MaxPooling2D, Deconvolution2D
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import keras
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import keras.backend.tensorflow_backend as K

print(keras.__version__)
H, W = 376, 312
def load_train_data():
        train_depth = np.load('./dataset/train_autoencoder_depth.npy')
        train_ir = np.load('./dataset/train_autoencoder_ir.npy')
        return train_depth, train_ir

def load_test_data():
        test_depth = np.load('./dataset/test_autoencoder_depth.npy')
        return test_depth



def coder_1layers(img_rows,img_cols):
    inputs = Input((img_rows, img_cols,1), name= 'input_layer')
    conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'conv1_1')(inputs)
    conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name= 'conv2_1')(pool1)
    conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name= 'conv2_2')(conv2)

    up1 = Deconvolution2D(64, 2, 2, output_shape=(None, 376, 312, 64), subsample=(2, 2), border_mode='same', name= 'deconv1')(conv2)
    up_conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'up_conv1_1')(up1)
    up_conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'up_conv1_2')(up_conv1)


    decoded = Conv2D(1, 3, 3, border_mode='same')(up_conv1)
    print(decoded.shape)

    model = Model(input=inputs, output=decoded)
    model.compile(loss="mse", optimizer='Adadelta')
    return model

def coder_2layers(img_rows,img_cols):
    inputs = Input((img_rows, img_cols,1), name= 'input_layer')
    conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'conv1_1')(inputs)
    conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name= 'conv2_1')(pool1)
    conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name= 'conv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1')(pool2)
    conv3 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2')(conv3)

    up2 = Deconvolution2D(128, 2, 2, output_shape=(None, 188, 156, 128), subsample=(2, 2), border_mode='same',name='deconv2')(conv3)
    up_conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='up_conv2_1')(up2)
    up_conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='up_conv2_2')(up_conv2)

    up1 = Deconvolution2D(64, 2, 2, output_shape=(None, 376, 312, 64), subsample=(2, 2), border_mode='same', name= 'deconv1')(up_conv2)
    up_conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'up_conv1_1')(up1)
    up_conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'up_conv1_2')(up_conv1)


    decoded = Conv2D(1, 3, 3, border_mode='same')(up_conv1)
    print(decoded.shape)

    model = Model(input=inputs, output=decoded)
    model.compile(loss="mse", optimizer='Adadelta')
    return model

def coder_3layers(img_rows,img_cols):
    inputs = Input((img_rows, img_cols,1), name= 'input_layer')
    conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'conv1_1')(inputs)
    conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name= 'conv2_1')(pool1)
    conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name= 'conv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1')(pool2)
    conv3 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1')(pool3)
    conv4 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2')(conv4)

    up3 = Deconvolution2D(256, 2, 2, output_shape=(None, 94, 78, 256), subsample=(2, 2), border_mode='same',name='deconv3')(conv4)
    up_conv3 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='up_conv3_1')(up3)
    up_conv3 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='up_conv3_2')(up_conv3)

    up2 = Deconvolution2D(128, 2, 2, output_shape=(None, 188, 156, 128), subsample=(2, 2), border_mode='same',name='deconv2')(up_conv3)
    up_conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='up_conv2_1')(up2)
    up_conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='up_conv2_2')(up_conv2)

    up1 = Deconvolution2D(64, 2, 2, output_shape=(None, 376, 312, 64), subsample=(2, 2), border_mode='same', name= 'deconv1')(up_conv2)
    up_conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'up_conv1_1')(up1)
    up_conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name= 'up_conv1_2')(up_conv1)


    decoded = Conv2D(1, 3, 3, border_mode='same')(up_conv1)
    print(decoded.shape)

    model = Model(input=inputs, output=decoded)
    model.compile(loss="mse", optimizer='Adadelta')
    return model

def unet(img_rows,img_cols):
    inputs = Input((img_rows, img_cols,1))
    conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(conv4)

    # up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = merge([Deconvolution2D(256, 2, 2, output_shape=(None, 94, 78, 256), subsample=(2, 2), border_mode='same')(conv4), conv3], mode='concat', concat_axis=3)
    up_conv1 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(up1)
    up_conv1 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(up_conv1)

    # up2 = UpSampling2D(size=(2, 2))(up_conv1)
    up2 = merge([Deconvolution2D(128, 2, 2, output_shape=(None, 188, 156, 128), subsample=(2, 2), border_mode='same')(up_conv1), conv2], mode='concat', concat_axis=3)
    up_conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(up2)
    up_conv2 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(up_conv2)

    # up3 = UpSampling2D(size=(2, 2))(up_conv2)
    up3 = merge([Deconvolution2D(64, 2, 2, output_shape=(None, 376, 312, 64), subsample=(2, 2), border_mode='same')(up_conv2), conv1], mode='concat', concat_axis=3)
    up_conv3 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up3)
    up_conv3 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up_conv3)

    decoded = Conv2D(1, 3, 3,border_mode='same')(up_conv3)
    print(decoded.shape)
    # optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    model = Model(input=inputs, output=decoded)
    model.compile(loss="mse", optimizer='adadelta')
    return model

def unet2(img_rows,img_cols):
    inputs = Input((1, img_rows, img_cols))
    conv1 = Conv2D(16, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(16, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    print(pool3.shape)
    conv4 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv4)

    # up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
    up_conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    up_conv1 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up_conv1)

    # up2 = UpSampling2D(size=(2, 2))(up_conv1)
    up2 = merge([UpSampling2D(size=(2, 2))(up_conv1), conv2], mode='concat', concat_axis=1)
    up_conv2 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    up_conv2 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up_conv2)

    # up3 = UpSampling2D(size=(2, 2))(up_conv2)
    up3 = merge([UpSampling2D(size=(2, 2))(up_conv2), conv1], mode='concat', concat_axis=1)
    up_conv3 = Conv2D(16, 3, 3, activation='relu', border_mode='same')(up3)
    up_conv3 = Conv2D(16, 3, 3, activation='relu', border_mode='same')(up_conv3)

    decoded = Conv2D(1, 3, 3, activation='sigmoid', border_mode='same')(up_conv3)
    print(decoded.shape)

    model = Model(input=inputs, output=decoded)
    model.compile(loss="categorical_crossentropy", optimizer='adadelta')
    return model

def train(model, weight_path):
    with K.tf.device('/gpu:3'):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))
        model = model
        train_depth, train_ir = load_train_data()
        train_depth = train_depth.astype('float32')
        train_depth = train_depth[:,:,:,0:1]
        mean = np.mean(train_depth)  # mean for data centering
        std = np.std(train_depth)  # std for data normalization
        train_depth -= mean
        train_depth /= std

        train_ir = train_ir.astype('float32') / 255.0
        train_ir = train_ir[:, :, :, 0:1]
        depth_set = np.ndarray((train_depth.shape[0]*3, train_depth.shape[1], train_depth.shape[2], 1))
        ir_set = np.ndarray((train_depth.shape[0] * 3, train_depth.shape[1], train_depth.shape[2], 1))

        i = 0
        j = 0
        while i < train_depth.shape[0]:
            M_left = cv2.getRotationMatrix2D((312 / 2, 376 / 2), -20, 1)
            M_right = cv2.getRotationMatrix2D((312 / 2, 376 / 2), 20, 1)
            train_depth_left = cv2.warpAffine(train_depth[i], M_left, (312, 376))
            train_depth_right = cv2.warpAffine(train_depth[i], M_right, (312, 376))
            train_ir_left = cv2.warpAffine(train_ir[i], M_left, (312, 376))
            train_ir_right = cv2.warpAffine(train_ir[i], M_right, (312, 376))
            depth_set[j] = train_depth[i]
            depth_set[j + 1,:,:,0] = train_depth_left
            depth_set[j + 2,:,:,0] = train_depth_right
            ir_set[j] = train_ir[i]
            ir_set[j + 1,:,:,0] = train_ir_left
            ir_set[j + 2,:,:,0] = train_ir_right
            i += 1
            j += 3

        print('extending done')
        print(depth_set.shape)
        print(ir_set.shape)

        print('Creating and compiling model...')
        print('-'*30)
        checkpointer = ModelCheckpoint(filepath=weight_path, verbose=1)#, save_best_only=True
        model.fit(depth_set, ir_set, batch_size=20, nb_epoch=30, verbose=1, shuffle=True, validation_split=0.2, callbacks=[checkpointer])

def test(model, weight_path, saved_result):
    with K.tf.device('/gpu:3'):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))
        model = model
        print('Loading and preprocessing train data...')
        print('-' * 30)
        imgs_test = load_test_data()
        imgs_test = imgs_test.astype('float32')
        imgs_test = imgs_test[:,:,:,0:1]

        mean = np.mean(imgs_test)  # mean for data centering
        std = np.std(imgs_test)  # std for data normalization
        imgs_test -= mean
        imgs_test /= std


        print('Loading saved weights...')
        print('-' * 30)
        model.load_weights(weight_path)
        print('Predicting masks on play data...')
        print('-' * 30)
        test_predict = model.predict(imgs_test, verbose=1,batch_size= 32)
        np.save(saved_result, test_predict*255)
        # test_predict = np.rollaxis(test_predict, 1, 4)

        i = 0
        for img_vis in test_predict:

            cv2.imwrite('result/demo/' + str(i) + '.png', img_vis*255)
            i += 1


def fake_train():

    model = autoencoder(H, W)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    depth , ir = load_train_data()
    imgs_test = depth[:, :, :, 0:1]
    imgs_test = np.rollaxis(imgs_test, 3, 1)
    imgs_test = imgs_test.astype('float32') / 255 * 4000
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization

    imgs_test -= mean
    imgs_test /= std
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('weights_autoencoder_old.hdf5')
    #     model.load_weights('best.hdf5')

    print('Predicting masks on play data...')
    print('-' * 30)
    test_predict = model.predict(imgs_test, verbose=1, batch_size=32)
    test_predict = np.rollaxis(test_predict, 1, 4)
    np.save('./dataset/faked_train.npy', test_predict*255)
    for image in test_predict:
        cv2.imshow('result', image)
        cv2.waitKey(500)

def fake_test():
    with K.tf.device('/gpu:2'):
        model = unet2(H, W)
        print('Loading and preprocessing train data...')
        print('-' * 30)
        depth = load_test_data()
        imgs_test = depth[:, :, :, 0:1]
        imgs_test = np.rollaxis(imgs_test, 3, 1)
        imgs_test = imgs_test.astype('float32') / 255 * 4000
        mean = np.mean(imgs_test)  # mean for data centering
        std = np.std(imgs_test)  # std for data normalization

        imgs_test -= mean
        imgs_test /= std
        print(imgs_test.shape)
        print('Loading saved weights...')
        print('-' * 30)
        model.load_weights('weights_autoencoder_old.hdf5')

        print('Predicting masks on play data...')
        print('-' * 30)
        test_predict = model.predict(imgs_test, verbose=1, batch_size=32)
        test_predict = np.rollaxis(test_predict, 1, 4)
        print(test_predict.shape)

        np.save('./dataset/faked_test.npy', test_predict*255)

        for f_img in test_predict:
            cv2.imshow('result', f_img)
            cv2.waitKey(500)


if __name__ == '__main__':
    weight_path = "model_keras/coder_3layers.hdf5"
    model = unet(H, W )
    train(model, weight_path)

    saved_result = './result/autoencoder3unet.npy'
    test(model, weight_path, saved_result)



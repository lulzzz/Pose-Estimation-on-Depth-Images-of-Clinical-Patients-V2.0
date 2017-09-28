from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

from dataset import Dataset_2, Dataset_3
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

H, W = 376, 312
train_depth_path = './dataset/train_autoencoder_depth.npy'
train_ir_path = './dataset/train_autoencoder_ir.npy'
test_depth_path = './dataset/test_autoencoder_depth.npy'
test_ir_path = './dataset/test_autoencoder_depth.npy'
batch_size = 20

def deepnn_1down(x_image):
  # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 64, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
  # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 128, 128])
        b_conv4 = bias_variable([128])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    #deconv layer
    with tf.name_scope('deconv1'):
        w_deconv1 = weight_variable([2, 2, 64, 128])
        b_deconv1 = bias_variable([64])
        h_deconv1 = tf.nn.relu(deconv2d(h_conv4, w_deconv1) + b_deconv1)

    with tf.name_scope('conv_up1'):
        W_conv_up1 = weight_variable([3, 3, 64, 64])
        b_conv_up1 = bias_variable([64])
        h_conv_up1 = tf.nn.relu(conv2d(h_deconv1, W_conv_up1) + b_conv_up1)
    with tf.name_scope('conv_up2'):
        W_conv_up2 = weight_variable([3, 3, 64, 64])
        b_conv_up2 = bias_variable([64])
        h_conv_up2 = tf.nn.relu(conv2d(h_conv_up1, W_conv_up2) + b_conv_up2)

    with tf.name_scope('output'):
        W_conv_output = weight_variable([1, 1, 64, 1])
        b_conv_output = bias_variable([1])
        output = conv2d(h_conv_up2, W_conv_output) + b_conv_output

    return output

def deepnn_2down(x_image):
  # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 64, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
  # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 128, 128])
        b_conv4 = bias_variable([128])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv4)


    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([3, 3, 128, 256])
        b_conv5 = bias_variable([256])
        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable([3, 3, 256, 256])
        b_conv6 = bias_variable([256])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    #deconv layer1
    with tf.name_scope('deconv1'):
        w_deconv1 = weight_variable([2, 2, 128, 256])
        b_deconv1 = bias_variable([128])
        h_deconv1 = tf.nn.relu(deconv2d(h_conv6, w_deconv1) + b_deconv1)

    with tf.name_scope('conv_up1'):
        W_conv_up1 = weight_variable([3, 3, 128, 128])
        b_conv_up1 = bias_variable([128])
        h_conv_up1 = tf.nn.relu(conv2d(h_deconv1, W_conv_up1) + b_conv_up1)
    with tf.name_scope('conv_up2'):
        W_conv_up2 = weight_variable([3, 3, 128, 128])
        b_conv_up2 = bias_variable([128])
        h_conv_up2 = tf.nn.relu(conv2d(h_conv_up1, W_conv_up2) + b_conv_up2)

    # deconv layer2
    with tf.name_scope('deconv2'):
        w_deconv2 = weight_variable([2, 2, 64, 128])
        b_deconv2 = bias_variable([64])
        h_deconv2 = tf.nn.relu(deconv2d(h_conv_up2, w_deconv2) + b_deconv2)

    with tf.name_scope('conv_up3'):
        W_conv_up3 = weight_variable([3, 3, 64, 64])
        b_conv_up3 = bias_variable([64])
        h_conv_up3 = tf.nn.relu(conv2d(h_deconv2, W_conv_up3) + b_conv_up3)
    with tf.name_scope('conv_up4'):
        W_conv_up4 = weight_variable([3, 3, 64, 64])
        b_conv_up4 = bias_variable([64])
        h_conv_up4 = tf.nn.relu(conv2d(h_conv_up3, W_conv_up4) + b_conv_up4)

    with tf.name_scope('output'):
        W_conv_output = weight_variable([1, 1, 64, 1])
        b_conv_output = bias_variable([1])
        output = conv2d(h_conv_up4, W_conv_output) + b_conv_output

    return output

def deepnn_3down(x_image):
  # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 64, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
  # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 128, 128])
        b_conv4 = bias_variable([128])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv4)


    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([3, 3, 128, 256])
        b_conv5 = bias_variable([256])
        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable([3, 3, 256, 256])
        b_conv6 = bias_variable([256])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv6)


    with tf.name_scope('conv7'):
        W_conv7 = weight_variable([3, 3, 256, 512])
        b_conv7 = bias_variable([512])
        h_conv7 = tf.nn.relu(conv2d(h_pool3, W_conv7) + b_conv7)
    with tf.name_scope('conv8'):
        W_conv8 = weight_variable([3, 3, 512, 512])
        b_conv8 = bias_variable([512])
        h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)

    #deconv layer1
    with tf.name_scope('deconv1'):
        w_deconv1 = weight_variable([2, 2, 256, 512])
        b_deconv1 = bias_variable([256])
        h_deconv1 = tf.nn.relu(deconv2d(h_conv8, w_deconv1) + b_deconv1)

    with tf.name_scope('conv_up1'):
        W_conv_up1 = weight_variable([3, 3, 256, 256])
        b_conv_up1 = bias_variable([256])
        h_conv_up1 = tf.nn.relu(conv2d(h_deconv1, W_conv_up1) + b_conv_up1)
    with tf.name_scope('conv_up2'):
        W_conv_up2 = weight_variable([3, 3, 256, 256])
        b_conv_up2 = bias_variable([256])
        h_conv_up2 = tf.nn.relu(conv2d(h_conv_up1, W_conv_up2) + b_conv_up2)

    # deconv layer2
    with tf.name_scope('deconv2'):
        w_deconv2 = weight_variable([2, 2, 128, 256])
        b_deconv2 = bias_variable([128])
        h_deconv2 = tf.nn.relu(deconv2d(h_conv_up2, w_deconv2) + b_deconv2)

    with tf.name_scope('conv_up3'):
        W_conv_up3 = weight_variable([3, 3, 128, 128])
        b_conv_up3 = bias_variable([128])
        h_conv_up3 = tf.nn.relu(conv2d(h_deconv2, W_conv_up3) + b_conv_up3)
    with tf.name_scope('conv_up4'):
        W_conv_up4 = weight_variable([3, 3, 128, 128])
        b_conv_up4 = bias_variable([128])
        h_conv_up4 = tf.nn.relu(conv2d(h_conv_up3, W_conv_up4) + b_conv_up4)

    # deconv layer3
    with tf.name_scope('deconv3'):
        w_deconv3 = weight_variable([2, 2, 64, 128])
        b_deconv3 = bias_variable([64])
        h_deconv3 = tf.nn.relu(deconv2d(h_conv_up4, w_deconv3) + b_deconv3)

    with tf.name_scope('conv_up5'):
        W_conv_up5 = weight_variable([3, 3, 64, 64])
        b_conv_up5 = bias_variable([64])
        h_conv_up5 = tf.nn.relu(conv2d(h_deconv3, W_conv_up5) + b_conv_up5)
    with tf.name_scope('conv_up6'):
        W_conv_up6 = weight_variable([3, 3, 64, 64])
        b_conv_up6 = bias_variable([64])
        h_conv_up6 = tf.nn.relu(conv2d(h_conv_up5, W_conv_up6) + b_conv_up6)

    with tf.name_scope('output'):
        W_conv_output = weight_variable([1, 1, 64, 1])
        b_conv_output = bias_variable([1])
        output = conv2d(h_conv_up6, W_conv_output) + b_conv_output

    return output

def deepnn_3down_unet(x_image):
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 64, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 128, 128])
        b_conv4 = bias_variable([128])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv4)

    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([3, 3, 128, 256])
        b_conv5 = bias_variable([256])
        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable([3, 3, 256, 256])
        b_conv6 = bias_variable([256])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv6)

    with tf.name_scope('conv7'):
        W_conv7 = weight_variable([3, 3, 256, 512])
        b_conv7 = bias_variable([512])
        h_conv7 = tf.nn.relu(conv2d(h_pool3, W_conv7) + b_conv7)
    with tf.name_scope('conv8'):
        W_conv8 = weight_variable([3, 3, 512, 512])
        b_conv8 = bias_variable([512])
        h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)

    # deconv layer1
    with tf.name_scope('deconv1'):
        w_deconv1 = weight_variable([2, 2, 256, 512])
        b_deconv1 = bias_variable([256])
        h_deconv1 = tf.nn.relu(deconv2d(h_conv8, w_deconv1) + b_deconv1)
        h_deconv_concat1 = tf.concat([h_conv6, h_deconv1], 3)

    with tf.name_scope('conv_up1'):
        W_conv_up1 = weight_variable([3, 3, 512, 256])
        b_conv_up1 = bias_variable([256])
        h_conv_up1 = tf.nn.relu(conv2d(h_deconv_concat1, W_conv_up1) + b_conv_up1)
    with tf.name_scope('conv_up2'):
        W_conv_up2 = weight_variable([3, 3, 256, 256])
        b_conv_up2 = bias_variable([256])
        h_conv_up2 = tf.nn.relu(conv2d(h_conv_up1, W_conv_up2) + b_conv_up2)

    # deconv layer2
    with tf.name_scope('deconv2'):
        w_deconv2 = weight_variable([2, 2, 128, 256])
        b_deconv2 = bias_variable([128])
        h_deconv2 = tf.nn.relu(deconv2d(h_conv_up2, w_deconv2) + b_deconv2)
        h_deconv_concat2 = tf.concat([h_conv4, h_deconv2], 3)

    with tf.name_scope('conv_up3'):
        W_conv_up3 = weight_variable([3, 3, 256, 128])
        b_conv_up3 = bias_variable([128])
        h_conv_up3 = tf.nn.relu(conv2d(h_deconv_concat2, W_conv_up3) + b_conv_up3)
    with tf.name_scope('conv_up4'):
        W_conv_up4 = weight_variable([3, 3, 128, 128])
        b_conv_up4 = bias_variable([128])
        h_conv_up4 = tf.nn.relu(conv2d(h_conv_up3, W_conv_up4) + b_conv_up4)

    # deconv layer3
    with tf.name_scope('deconv3'):
        w_deconv3 = weight_variable([2, 2, 64, 128])
        b_deconv3 = bias_variable([64])
        h_deconv3 = tf.nn.relu(deconv2d(h_conv_up4, w_deconv3) + b_deconv3)
        h_deconv_concat3 = tf.concat([h_conv2, h_deconv3], 3)

    with tf.name_scope('conv_up5'):
        W_conv_up5 = weight_variable([3, 3, 128, 64])
        b_conv_up5 = bias_variable([64])
        h_conv_up5 = tf.nn.relu(conv2d(h_deconv_concat3, W_conv_up5) + b_conv_up5)
    with tf.name_scope('conv_up6'):
        W_conv_up6 = weight_variable([3, 3, 64, 64])
        b_conv_up6 = bias_variable([64])
        h_conv_up6 = tf.nn.relu(conv2d(h_conv_up5, W_conv_up6) + b_conv_up6)

    with tf.name_scope('output'):
        W_conv_output = weight_variable([1, 1, 64, 1])
        b_conv_output = bias_variable([1])
        output = conv2d(h_conv_up6, W_conv_output) + b_conv_output

    return output

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def deconv2d(x, W):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def rotate_image(image_set):
    rotated_image_set = np.ndarray((image_set.shape[0] * 3, image_set.shape[1], image_set.shape[2], 1), dtype=np.float32)
    M_left = cv2.getRotationMatrix2D((image_set.shape[2] / 2, image_set.shape[1] / 2), 20, 1)
    M_right = cv2.getRotationMatrix2D((image_set.shape[2] / 2, image_set.shape[1] / 2), -20, 1)
    j = 0
    for i in range(image_set.shape[0]):
        image_left = cv2.warpAffine(image_set[i], M_left, (image_set.shape[2], image_set.shape[1]))
        image_right = cv2.warpAffine(image_set[i], M_right, (image_set.shape[2], image_set.shape[1]))

        rotated_image_set[j] = image_set[i]
        rotated_image_set[j + 1,:,:,0] = image_left
        rotated_image_set[j + 2,:,:,0] = image_right
        j += 3
    return rotated_image_set

def make_summary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

def train(saved_coder_path,tensorboard_path):
    with tf.device('/gpu:2'):
        tf_config = tf.ConfigProto(log_device_placement=True)
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        x = tf.placeholder(tf.float32, [None, H, W, 1])

        # Define loss and optimizer
        y_true = tf.placeholder(tf.float32, [None, H, W, 1])

        # Build the graph for the deep net
        y_predict = deepnn_1down(x)

        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.pow(y_true - y_predict, 2))

        tf.summary.scalar('train_cost', cost)
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        summary_op = tf.summary.merge_all()

        with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost, global_step=global_step)



        train_depth = np.load(train_depth_path)
        train_ir = np.load(train_ir_path)
        train_depth = train_depth.astype('float32')
        train_depth = train_depth[:, :, :, 0:1]
        mean = np.mean(train_depth)  # mean for data centering
        std = np.std(train_depth)  # std for data normalization
        train_depth -= mean
        train_depth /= std

        train_ir = train_ir.astype('float32') / 255.0
        train_ir = train_ir[:, :, :, 0:1]
        X_train, validation_set, y_train, annotation_validation_set = train_test_split(train_depth, train_ir, test_size=0.2, random_state=42)

        train_set = rotate_image(X_train)
        annotation_train_set = rotate_image(y_train)
        print(train_set.shape)
        print(annotation_train_set.shape)

        dataset_train = Dataset_2(train_set, annotation_train_set)

        saver = tf.train.Saver()
        total_batch_train = int(len(train_set) / batch_size)
        total_batch_validation = int(len(validation_set) / batch_size)

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
            best_loss = 0
            for i in range(30000):
                if i % 200 == 0:
                    cost_sum = 0
                    for j in range(total_batch_validation):
                        if j < total_batch_validation:
                            batch_im_val = validation_set[j * batch_size:(j + 1) * batch_size]
                            batch_an_val = annotation_validation_set[j * batch_size:(j + 1) * batch_size]
                        else:
                            batch_im_val = validation_set[j * batch_size:]
                            batch_an_val = annotation_validation_set[j * batch_size:]
                        c_validation, step = sess.run([cost, global_step], feed_dict={x: batch_im_val, y_true: batch_an_val})
                        cost_sum = cost_sum + (c_validation * batch_size)
                    validation_cost = cost_sum / len(validation_set)

                    val_summary = make_summary('validation_cost', validation_cost)
                    writer.add_summary(val_summary, global_step=step)

                    print("validation cost=", "{:.9f}".format(validation_cost), " best validation loss=", "{:.9f}".format(best_loss))

                    if best_loss == 0:
                        best_loss = validation_cost
                    elif best_loss - validation_cost > 0:
                        best_loss = validation_cost
                        saver.save(sess, saved_coder_path)
                        print('update model')


                batch_im_train, batch_an_train = dataset_train.next_batch(batch_size)
                _, c_train, summary, step = sess.run([optimizer, cost, summary_op, global_step],
                                                     feed_dict={x: batch_im_train, y_true: batch_an_train})
                writer.add_summary(summary, global_step=step)
                print('batch: ' + str(i) + '/' + str(total_batch_train), '---', "train cost=", "{:.9f}".format(c_train))




if __name__ == '__main__':
    saved_coder_path = 'CPM_model/coder_1_layer_new.ckpt'
    tensorboard_path = 'tensorboard_coder/coder_1_layer_new/'
    train(saved_coder_path,tensorboard_path)





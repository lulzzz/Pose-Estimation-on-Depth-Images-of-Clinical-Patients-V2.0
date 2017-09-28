from __future__ import print_function
from __future__ import division

import numpy as np
from dataset import Dataset_2, Dataset_3
import skimage.io
import skimage.transform
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import cv2
import tensorflow as tf
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import cpm


    
def detect_objects_heatmap(heatmap):
    data = 256 * heatmap 
    data_max = filters.maximum_filter(data, 3, mode='reflect')
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, 3, mode='reflect')
    diff = ((data_max - data_min) > 0.3)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    objects = np.zeros((num_objects, 2), dtype=np.int32)
    for oid,(dy,dx) in enumerate(slices):
        objects[oid,:] = [(dy.start + dy.stop - 1)/2, (dx.start + dx.stop - 1)/2]
    return objects

def gaussian_kernel(h, w, sigma_h, sigma_w):
    yx = np.mgrid[-h//2:h//2,-w//2:w//2]**2
    return np.exp(-yx[0,:,:] / sigma_h**2 - yx[1,:,:] / sigma_w**2)

def prepare_input_posenet(image, objects, size_person, size, sigma=25, max_num_objects=3, border=400):
    result = np.zeros((max_num_objects, size[0], size[1], 4))
    padded_image = np.zeros((size_person[0]+border,size_person[1]+border,4))
    padded_image[border//2:-border//2,border//2:-border//2,:3] = image
    assert len(objects) < max_num_objects
    for oid, (yc, xc) in enumerate(objects):
        dh, dw = size[0]//2, size[1]//2
        y0, x0, y1, x1 = np.array([yc-dh, xc-dw, yc+dh, xc+dw]) + border//2
        result[oid,:,:,:4] = padded_image[y0:y1,x0:x1,:]
        result[oid,:,:,3] = gaussian_kernel(size[0], size[1], sigma, sigma)
    return np.split(result, [3], 3)

def detect_parts_heatmaps(heatmaps, centers, size, num_parts=14):
    parts = np.zeros((len(centers), num_parts, 2), dtype=np.int32)
    for oid, (yc, xc) in enumerate(centers):
        part_hmap = skimage.transform.resize(np.clip(heatmaps[oid], -1, 1), size, mode='reflect') 
        for pid in range(num_parts):
            y, x = np.unravel_index(np.argmax(part_hmap[:,:,pid]), size)
            parts[oid,pid] = y+yc-size[0]//2,x+xc-size[1]//2
    return parts

def gen_kernel(score_map, border=400, sigma_h = 25, sigma_w = 25):
    kernal = gaussian_kernel(score_map.shape[0]+border, score_map.shape[1]+border, sigma_h, sigma_w)
    y, x = np.unravel_index(np.argmax(score_map), [len(score_map), len(score_map[0])])
    dh, dw = score_map.shape[0] // 2, score_map.shape[1] // 2
    y0, x0, y1, x1 = np.array([dh - y, dw - x, 3*dh - y, 3*dw - x]) + border // 2
    return kernal[y0:y1, x0:x1]

def prepare_centermaps(train_centers, PH, PW, train_depth):
    single_center_maps = np.ndarray((PH, PW, 1), dtype=np.float32)
    single_centered_resized_maps = np.ndarray((47, 39, 1), dtype=np.float32)
    center_maps = np.ndarray((len(train_centers), 47, 39, 1), dtype=np.float32)
    for mid in range(len(train_centers)):
        score_map = np.zeros((PH, PW))
        score_map[train_centers[mid][0]][train_centers[mid][1]] = 1
        single_center_maps[:, :, 0] = gen_kernel(score_map)
        # if mid%2 == 0:
        #     cv2.imshow('result/demo', single_center_maps[:, :, 0] * 0.5 + train_depth[mid, :, :, 0] * 0.5)
        #     cv2.waitKey(100)
        single_centered_resized_maps = skimage.transform.resize(single_center_maps, (47,39), mode='reflect')
        center_maps[mid] = single_centered_resized_maps
    return center_maps

def find_centers(heatmaps, size):
    centers = np.zeros((len(heatmaps), 2), dtype=np.int16)
    for i in range(len(heatmaps)):
        y, x = np.unravel_index(np.argmax(heatmaps[i][:, :, 0]), size)
        centers[i] = y, x
    return centers

def calculate_distance(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5

def evaluation(truth_centers_coordinates, centers, parts_truth_coordinates, threshold, num_parts=8):
    predicted_parts_coordinates = centers.astype('float16')
    matrix = [0,0]
    for mid in range(centers.shape[0]):
        standard_distance = threshold * calculate_distance(parts_truth_coordinates[mid][0], parts_truth_coordinates[mid][1])
        distance = calculate_distance(truth_centers_coordinates[mid],predicted_parts_coordinates[mid])
        if distance <= standard_distance:
            matrix[0] += 1
            matrix[1] += 1
        else:
            matrix[1] += 1
    return matrix

learning_rate = 0.01
training_epochs = 20
batch_size = 20
display_step = 1
examples_to_show = 10


def CPM_pl(train_depth,train_centers):
    with tf.device('/gpu:2'):
        model_path = 'CPM_model/'
        person_net_path = os.path.join(model_path, 'person_net.ckpt')
        tf.reset_default_graph()
        #==================================================================================================================
        with tf.variable_scope('CPM'):
            # input dims for the person network
            PH, PW = 376, 312
            image_in = tf.placeholder(tf.float32, [None,PH,PW,3])
            heatmap_person, heatmap_stage3, heatmap_stage2, heatmap_stage1 = cpm.inference_person(image_in)
            # heatmap_person_large = tf.image.resize_images(heatmap_person, [PH, PW])
            # heatmap_stage3_large = tf.image.resize_images(heatmap_stage3, [PH, PW])
            # heatmap_stage2_large = tf.image.resize_images(heatmap_stage2, [PH, PW])
            # heatmap_stage1_large = tf.image.resize_images(heatmap_stage1, [PH, PW])
            # Prediction
            y_stage4 = heatmap_person
            y_stage3 = heatmap_stage3
            y_stage2 = heatmap_stage2
            y_stage1 = heatmap_stage1
            # Targets (Labels) are the input data.
            y_true = tf.placeholder(tf.float32, [None,47,39,1])
            # Define loss and optimizer, minimize the squared error
            cost = tf.reduce_mean(tf.pow(y_true - y_stage4, 2))\
                   + tf.reduce_mean(tf.pow(y_true - y_stage3, 2))\
                   + tf.reduce_mean(tf.pow(y_true - y_stage2, 2))\
                   + tf.reduce_mean(tf.pow(y_true - y_stage1, 2))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            tf_config.allow_soft_placement = True

            train_centers = prepare_centermaps(train_centers, PH, PW, train_depth)
            restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'CPM/PersonNet'))
            saver = tf.train.Saver()

            b_image = train_depth - 0.5
            dataset = Dataset_2(b_image, train_centers)
            print(heatmap_person.shape)
            print('match')
            print(train_centers.shape)
            init = tf.initialize_all_variables()
            with tf.Session(config=tf_config) as sess:
                sess.run(init)
                restorer.restore(sess, person_net_path)
                total_batch = int(len(train_depth) / batch_size)
                for epoch in range(training_epochs):
                    for i in range(total_batch):
                        print('batch: ' + str(i))
                        batch_xs, batch_ys = dataset.next_batch(batch_size)
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c ,hmap_person=sess.run([optimizer, cost, heatmap_person], feed_dict={ image_in : batch_xs, y_true: batch_ys})


                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
                    saver.save(sess, 'CPM_model/E0_ir.ckpt')
            print('done detecting')

def test(test_depth,test_centers, test_annotation):
    with tf.device('/gpu:2'):
        model_path = 'CPM_model/'
        person_net_path = os.path.join(model_path, 'E0_ir.ckpt')
        tf.reset_default_graph()
        with tf.variable_scope('CPM'):
            # input dims for the person network
            PH, PW = 376, 312
            image_in = tf.placeholder(tf.float32, [None,PH,PW,3])
            heatmap_person, heatmap_stage3, heatmap_stage2, heatmap_stage1 = cpm.inference_person(image_in)
            heatmap_person_large = tf.image.resize_images(heatmap_person, [PH, PW])

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        saver = tf.train.Saver()
        b_image = test_depth - 0.5
        result = []
        with tf.Session(config=tf_config) as sess:
            saver.restore(sess, person_net_path)
            total_batch = int(len(test_depth) / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                if i < total_batch - 1:
                    batch_xs = b_image[i * batch_size:(i + 1) * batch_size]
                else:
                    batch_xs = b_image[i * batch_size:]
                # Run optimization op (backprop) and cost op (to get loss value)
                hmap_person = sess.run(heatmap_person_large, feed_dict={image_in: batch_xs})
                for center_map in hmap_person:
                    result.append(center_map)
        print('done detecting')
        centers = find_centers(result, [PH, PW])

        matrix_1 = evaluation(centers, test_centers, test_annotation, 0.1, num_parts=8)
        matrix_2 = evaluation(centers, test_centers, test_annotation, 0.2, num_parts=8)
        matrix_3 = evaluation(centers, test_centers, test_annotation, 0.3, num_parts=8)
        matrix_4 = evaluation(centers, test_centers, test_annotation, 0.4, num_parts=8)
        matrix_5 = evaluation(centers, test_centers, test_annotation, 0.5, num_parts=8)
        matrix_6 = evaluation(centers, test_centers, test_annotation, 0.6, num_parts=8)
        matrix_7 = evaluation(centers, test_centers, test_annotation, 0.7, num_parts=8)
        matrix_8 = evaluation(centers, test_centers, test_annotation, 0.8, num_parts=8)
        matrix_9 = evaluation(centers, test_centers, test_annotation, 0.9, num_parts=8)
        matrix_10 = evaluation(centers, test_centers, test_annotation, 1, num_parts=8)
        print('matrix_1 -------------------------------------------------------------')
        print(matrix_1)
        print('matrix_2 -------------------------------------------------------------')
        print(matrix_2)
        print('matrix_3 -------------------------------------------------------------')
        print(matrix_3)
        print('matrix_4 -------------------------------------------------------------')
        print(matrix_4)
        print('matrix_5 -------------------------------------------------------------')
        print(matrix_5)
        print('matrix_6 -------------------------------------------------------------')
        print(matrix_6)
        print('matrix_7 -------------------------------------------------------------')
        print(matrix_7)
        print('matrix_8 -------------------------------------------------------------')
        print(matrix_8)
        print('matrix_9 -------------------------------------------------------------')
        print(matrix_9)
        print('matrix_10 -------------------------------------------------------------')
        print(matrix_10)

        for i in range(len(result)):
            # print(result[i][250])
            show = result[i] * 0.8 + test_depth[i, :, :, 0:1] * 0.2
            cv2.imshow('c', show)
            cv2.waitKey(500)


train_merged_path = './dataset/merged_train.npy'
train_depth_path = './dataset/train_autoencoder_depth.npy'
train_center_path = './dataset/train_autoencoder_center.npy'

test_merged_path = './dataset/merged_test.npy'
test_depth_path = './dataset/test_autoencoder_depth.npy'
test_center_path = './dataset/test_autoencoder_center.npy'

def load_person_train():
    train_depth = np.load('./dataset/train_autoencoder_ir.npy')
    train_centers = np.load(train_center_path)
    return train_depth, train_centers

def load_person_test():
    test_depth = np.load('./dataset/test_autoencoder_ir.npy')
    test_centers = np.load(test_center_path)
    return test_depth, test_centers

if __name__ == '__main__':
    train_images, train_centers= load_person_train()
    train_images = train_images.astype('float32')/255
    test_depth, test_centers = load_person_test()
    test_depth = test_depth.astype('float32') / 255
    test_annotation = np.load('./dataset/test_autoencoder_annotation.npy')
    # CPM_pl(train_images, train_centers)
    test(test_depth,test_centers,test_annotation)






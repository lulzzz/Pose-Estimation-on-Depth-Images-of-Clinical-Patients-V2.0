from __future__ import print_function
from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
from dataset import Dataset_2, Dataset_3
from sklearn.model_selection import train_test_split
import skimage.io
import skimage.transform
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import cv2
import tensorflow as tf

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
    objects = np.zeros((num_objects, 2), dtype=np.int16)
    for oid, (dy, dx) in enumerate(slices):
        objects[oid, :] = [(dy.start + dy.stop - 1) / 2, (dx.start + dx.stop - 1) / 2]
    return objects

def gaussian_kernel(h, w, sigma_h, sigma_w):
    yx = np.mgrid[-h // 2:h // 2, -w // 2:w // 2] ** 2
    return np.exp(-yx[0, :, :] / sigma_h ** 2 - yx[1, :, :] / sigma_w ** 2)


def detect_parts_heatmaps(heatmaps, centers, size, num_parts=14):
    parts = np.zeros((len(heatmaps), num_parts, 2), dtype=np.int16)
    for i in range(len(heatmaps)):
        part_hmap = skimage.transform.resize(np.clip(heatmaps[i], -1, 1), size, mode='reflect')
        for pid in range(num_parts):
            y, x = np.unravel_index(np.argmax(part_hmap[:, :, pid]), size)
            parts[i, pid] = y + centers[i][0] - size[0] // 2, x + centers[i][1] - size[1] // 2
    return parts


LIMBS = np.array([1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14]).reshape((-1, 2)) - 1
COLORS = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
          [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]]


def draw_limbs(image, parts):
    for lid, (p0, p1) in enumerate(LIMBS):
        y0, x0 = parts[p0]
        y1, x1 = parts[p1]
        cv2.circle(image, (x0, y0), 5, (0, 0, 0), -1)
        cv2.circle(image, (x1, y1), 5, (0, 0, 0), -1)
        cv2.line(image, (x0, y0), (x1, y1), COLORS[lid], 3)


def map_to_coordinates(annotations, size, num_parts=14):
    parts_coordinates = np.zeros((len(annotations), 14, 2), dtype=np.float16)
    for mid in range(len(annotations)):
        image_resized = skimage.transform.resize(annotations[mid], size, mode='reflect')
        for pid in range(num_parts):
            y, x = np.unravel_index(np.argmax(image_resized[:, :, pid]),
                                    [image_resized.shape[0], image_resized.shape[1]])
            parts_coordinates[mid][pid][0] = y
            parts_coordinates[mid][pid][1] = x
    return parts_coordinates


def calculate_distance(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def evaluation(annotations, parts, num_parts=8):
    parts_truth_coordinates = map_to_coordinates(annotations, [376, 312])
    parts_predicted_coordinates = parts.astype('float16')
    matrix = np.zeros((num_parts, 2), dtype=np.uint16)
    wrong = []
    for mid in range(parts_truth_coordinates.shape[0]):
        standard_distance = 0.5 * calculate_distance(parts_truth_coordinates[mid][0], parts_truth_coordinates[mid][1])
        for pid in range(num_parts):
            distance = calculate_distance(parts_truth_coordinates[mid][pid], parts_predicted_coordinates[mid][pid])
            if distance <= standard_distance:
                matrix[pid][0] += 1
                matrix[pid][1] += 1
            else:
                wrong.append(mid)
                matrix[pid][1] += 1
    return matrix


def centering_image(images, centers, border=400 ):
    padded_image = np.zeros((images.shape[1] + border, images.shape[2] + border, images.shape[3]))
    results = np.zeros((images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
    dh, dw = images.shape[1] // 2, images.shape[2] // 2
    for i in range(len(images)):
        padded_image[border // 2:-border // 2, border // 2:-border // 2, :] = images[i]
        yc, xc = centers[i]
        y0, x0, y1, x1 = np.array([yc - dh, xc - dw, yc + dh, xc + dw]) + border // 2
        results[i, :, :, :] = padded_image[y0:y1, x0:x1, :]
    return results

def gen_kernel(score_map,border=400, sigma_h = 10, sigma_w = 10):
    kernal = gaussian_kernel(score_map.shape[0]+border, score_map.shape[1]+border, sigma_h, sigma_w)
    y, x = np.unravel_index(np.argmax(score_map), [len(score_map), len(score_map[0])])
    dh, dw = score_map.shape[0] // 2, score_map.shape[1] // 2
    y0, x0, y1, x1 = np.array([dh - y, dw - x, 3*dh - y, 3*dw - x]) + border // 2
    return kernal[y0:y1, x0:x1]

def prepare_centered_annotation(annotations, centers, H, W, num_parts=14 ):
    print(annotations.shape)
    print(centers.shape)
    single_annotation_maps = np.ndarray((1, H, W, 14), dtype=np.float32)
    single_centered_resized_maps = np.ndarray((46, 38, 15), dtype=np.float32)
    annotation_maps = np.ndarray((len(annotations), 46, 38, 15), dtype=np.float32)
    for mid in range(len(annotations)):
        for pid in range(num_parts):
            score_map = np.zeros((H, W))
            score_map[annotations[mid][pid][0]][annotations[mid][pid][1]] = 1
            single_annotation_maps[0,:,:,pid] = gen_kernel(score_map)
        # cv2.imwrite('help/0.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0,:,:,0]*0.3))
        # cv2.imwrite('help/1.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 1] * 0.3))
        # cv2.imwrite('help/2.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 2] * 0.3))
        # cv2.imwrite('help/3.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 3] * 0.3))
        # cv2.imwrite('help/4.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 4] * 0.3))
        # cv2.imwrite('help/5.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 5] * 0.3))
        # cv2.imwrite('help/6.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 6] * 0.3))
        # cv2.imwrite('help/7.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 7] * 0.3))
        # cv2.imwrite('help/8.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 8] * 0.3))
        # cv2.imwrite('help/9.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 9] * 0.3))
        # cv2.imwrite('help/10.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 10] * 0.3))
        # cv2.imwrite('help/11.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 11] * 0.3))
        # cv2.imwrite('help/12.png', 255*(train_depth[mid, :, :, 0] * 0.7 + single_annotation_maps[0, :, :, 12] * 0.3))
        # cv2.imwrite('help/13.png', 255*(train_depth[mid, :, :, 0] * 0.7+ single_annotation_maps[0, :, :, 13] * 0.3))

        single_centered_maps = centering_image(single_annotation_maps,[centers[mid]])
        single_centered_resized_maps[:,:,:14] = skimage.transform.resize(single_centered_maps[0], (46,38), mode='reflect')
        score_map_14 = np.ones((46, 38))
        for pid in range(0, 14):
            score_map_14 -= single_centered_resized_maps[:, :, pid]
            single_centered_resized_maps[:, :, 14] = score_map_14
        annotation_maps[mid] = single_centered_resized_maps

        # cv2.imshow('map',annotation_maps[mid,:,:,14])
        # cv2.waitKey(500)
    return annotation_maps

def rotate_image(image_set):
    rotated_image_set = np.ndarray((image_set.shape[0] * 3, image_set.shape[1], image_set.shape[2], 3), dtype=np.float32)
    M_left = cv2.getRotationMatrix2D((image_set.shape[2] / 2, image_set.shape[1] / 2), 20, 1)
    M_right = cv2.getRotationMatrix2D((image_set.shape[2] / 2, image_set.shape[1] / 2), -20, 1)
    j = 0
    for i in range(image_set.shape[0]):
        image_left = cv2.warpAffine(image_set[i], M_left, (image_set.shape[2], image_set.shape[1]))
        image_right = cv2.warpAffine(image_set[i], M_right, (image_set.shape[2], image_set.shape[1]))

        rotated_image_set[j] = image_set[i]
        rotated_image_set[j + 1,] = image_left
        rotated_image_set[j + 2,] = image_right

        #
        # cv2.imshow('1',rotated_image_set[j])
        # cv2.imshow('2', rotated_image_set[j+1])
        # cv2.imshow('3', rotated_image_set[j+2])
        # cv2.waitKey(500)
        j += 3
    return rotated_image_set

def rotate_annotation(annotation_set):
    rotated_annotation_set = np.ndarray((annotation_set.shape[0] * 3, annotation_set.shape[1], annotation_set.shape[2], 15), dtype=np.float32)
    M_left = cv2.getRotationMatrix2D((annotation_set.shape[2] / 2, annotation_set.shape[1] / 2), 20, 1)
    M_right = cv2.getRotationMatrix2D((annotation_set.shape[2] / 2, annotation_set.shape[1] / 2), -20, 1)
    j = 0
    for i in range(annotation_set.shape[0]):
        image_left = cv2.warpAffine(annotation_set[i], M_left, (annotation_set.shape[2], annotation_set.shape[1]))
        image_right = cv2.warpAffine(annotation_set[i], M_right, (annotation_set.shape[2], annotation_set.shape[1]))
        rotated_annotation_set[j] = annotation_set[i]
        rotated_annotation_set[j + 1, :, :, :14] = image_left[:, :, :14]
        rotated_annotation_set[j + 2, :, :, :14] = image_right[:, :, :14]
        score_map_14_left = np.ones((annotation_set.shape[1], annotation_set.shape[2]))
        score_map_14_right = np.ones((annotation_set.shape[1], annotation_set.shape[2]))
        for pid in range(0, 14):
            score_map_14_left -= image_left[:, :, pid]
            score_map_14_right -= image_right[:, :, pid]
        rotated_annotation_set[j + 1, :, :, 14] = score_map_14_left
        rotated_annotation_set[j + 2, :, :, 14] = score_map_14_right

        # cv2.imshow('1', rotated_annotation_set[j + 1, :, :, 10])
        # cv2.imshow('2', rotated_annotation_set[j + 1, :, :, 14])
        # cv2.imshow('3', rotated_annotation_set[j + 2, :, :, 14])
        # cv2.waitKey(500)
        j += 3
    return rotated_annotation_set

learning_rate = 0.01
training_epochs = 10
batch_size = 16
display_step = 1
examples_to_show = 10

def make_summary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

def CPM(train_depth, train_centers, train_annotation, pretrained_cpm_path, saved_cpm_path, tensorboard_path):

    with tf.device('/gpu:1'):
        pose_net_path = pretrained_cpm_path
        tf.reset_default_graph()

        with tf.variable_scope('CPM'):
            # input dims for the pose network
            N, H, W = 3, 376, 312
            sH, sW, = 46, 38
            pose_image_in = tf.placeholder(tf.float32, [None, H, W, 3])
            pose_centermap_in = tf.placeholder(tf.float32, [None, H, W, 1])
            heatmap_pose, heatmap_stage5, heatmap_stage4, heatmap_stage3, heatmap_stage2, heatmap_stage1= cpm.inference_pose(pose_image_in, pose_centermap_in)

        tf_config = tf.ConfigProto(log_device_placement=True)
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        # Prediction
        y_stage6 = heatmap_pose
        y_stage5 = heatmap_stage5
        y_stage4 = heatmap_stage4
        y_stage3 = heatmap_stage3
        y_stage2 = heatmap_stage2
        y_stage1 = heatmap_stage1
        # Targets (Labels) are the input data.
        y_true = tf.placeholder(tf.float32, [None, sH, sW, 15])
        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true[:,:,:,0:9] - y_stage6[:,:,:,0:9], 2)) \
               + tf.reduce_mean(tf.pow(y_true[:,:,:,0:9] - y_stage5[:,:,:,0:9], 2)) \
               + tf.reduce_mean(tf.pow(y_true[:,:,:,0:9] - y_stage4[:,:,:,0:9], 2)) \
               + tf.reduce_mean(tf.pow(y_true[:,:,:,0:9] - y_stage3[:,:,:,0:9], 2)) \
               + tf.reduce_mean(tf.pow(y_true[:,:,:,0:9] - y_stage2[:,:,:,0:9], 2)) \
               + tf.reduce_mean(tf.pow(y_true[:,:,:,0:9] - y_stage1[:,:,:,0:9], 2))

        tf.summary.scalar('train_cost',cost)
        global_step = tf.Variable(initial_value=0,trainable=False,name='global_step')
        learning_rate = 0.01
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        summary_op = tf.summary.merge_all()

        # centered depth images
        centered_images = centering_image(train_depth, train_centers)

        # centered_annotations = prepare_centered_annotation(train_annotation, train_centers, H, W)
        # np.save('./dataset/train_autoencoder_depth_centered.npy', centered_annotations)
        centered_annotations = np.load('./dataset/train_autoencoder_depth_centered.npy')
        X_train, validation_set, y_train, annotation_validation_set = train_test_split(centered_images, centered_annotations, test_size = 0.2, random_state = 42)
        print('training size:' + str(len(X_train)))
        print('validation size:' + str(len(validation_set)))

        #rotate images
        train_set = rotate_image(X_train)
        annotation_train_set = rotate_annotation(y_train)
        print(train_set.shape)
        print(annotation_train_set.shape)

        # generate kernels
        kernels = np.zeros((validation_set.shape[0], H, W, 1))
        for i in range(kernels.shape[0]):
            kernels[i, :, :, 0] = gaussian_kernel(H, W, 25, 25)


        b_image_train = train_set - 0.5
        b_image_validation = validation_set - 0.5
        dataset_train= Dataset_2(b_image_train, annotation_train_set)

        restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'CPM/PoseNet'))
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        total_batch_train = int(len(train_set) / batch_size)
        total_batch_validation = int(len(validation_set) / batch_size)
        with tf.Session(config=tf_config) as sess:
            sess.run(init)
            restorer.restore(sess, pose_net_path)
            writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
            #validation
            best_loss = 0
            for i in range(30000):

                if i % 200 == 0:
                    cost_sum = 0
                    for j in range(total_batch_validation):
                        if j < total_batch_validation - 1:
                            batch_im_val = b_image_validation[j * batch_size:(j + 1) * batch_size]
                            batch_an_val = annotation_validation_set[j * batch_size:(j + 1) * batch_size]
                            batch_kernel_val = kernels[j * batch_size:(j + 1) * batch_size]
                        else:
                            batch_im_val = b_image_validation[j * batch_size:]
                            batch_an_val = annotation_validation_set[j * batch_size:]
                            batch_kernel_val = kernels[j * batch_size:]
                        c_validation, step = sess.run([cost, global_step], feed_dict={pose_image_in: batch_im_val,
                                                                   pose_centermap_in: batch_kernel_val,
                                                                   y_true: batch_an_val})
                        cost_sum = cost_sum + (c_validation * batch_size)
                    validation_cost = cost_sum / len(b_image_validation)

                    val_summary = make_summary('validation_cost', validation_cost)
                    writer.add_summary(val_summary, global_step=step)

                    print("validation cost=", "{:.9f}".format(validation_cost), " best validation loss=",
                          "{:.9f}".format(best_loss), " learning rate=" + str(learning_rate))

                    if best_loss == 0:
                        best_loss = validation_cost
                    elif best_loss - validation_cost > 0.000005:
                        best_loss = validation_cost
                        saver.save(sess, saved_cpm_path)
                        print('update model')
                    elif  best_loss - validation_cost <= 0.000005 and learning_rate > 0.001:
                        learning_rate *= 0.5
                        print('learning rate model')


                batch_im_train, batch_an_train = dataset_train.next_batch(batch_size)
                batch_kernel_train = kernels[:batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c_train, summary, step = sess.run([optimizer, cost, summary_op, global_step], feed_dict={pose_image_in: batch_im_train,
                                                                    pose_centermap_in: batch_kernel_train,
                                                                    y_true: batch_an_train})
                writer.add_summary(summary, global_step=step)
                print('batch: ' + str(i) + '/' + str(total_batch_train), '---',"train cost=", "{:.9f}".format(c_train), " learning rate=" + str(learning_rate))

        print('done detecting')

train_depth_path = './dataset/train_autoencoder_depth.npy'
train_ir_path = './dataset/train_autoencoder_ir.npy'
train_merge_path = './dataset/merged_train.npy'
train_center_path = './dataset/train_autoencoder_center.npy'
train_annotation_path = './dataset/train_autoencoder_annotation.npy'


if __name__ == '__main__':
    train_images = np.load(train_merge_path)
    train_centers = np.load(train_center_path)
    train_annotation = np.load(train_annotation_path)
    train_images = train_images.astype('float32')/255.0

    pretrained_cpm_path = 'CPM_model/pose_net.ckpt'
    saved_cpm_path = 'CPM_model/pose_net_03.ckpt'
    tensorboard_path = 'tensorboard/merge_03/'
    CPM(train_images, train_centers, train_annotation, pretrained_cpm_path, saved_cpm_path, tensorboard_path)








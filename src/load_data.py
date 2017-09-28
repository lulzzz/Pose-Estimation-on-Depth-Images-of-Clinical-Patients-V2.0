import os
import numpy as np
import cv2
import json

start_index = 0
image_path = 'images/trial_3_autoencoder/'
test_path = 'test/trial_3_autoencoder/'
json_file = 'annotations/all_patient.json'

image_rows = 376
image_cols = 312
image_rows_map = 46
image_cols_map  = 38
with open(json_file) as jf:
    dict = json.load(jf)


def gaussian_kernel(h, w, sigma_h, sigma_w):
    yx = np.mgrid[-h//2:h//2,-w//2:w//2]**2
    return np.exp(-yx[0,:,:] / sigma_h**2 - yx[1,:,:] / sigma_w**2)

def max(a,b):
    return a if a>=b else b

def min(a,b):
    return a if a<=b else b

def gen_kernel(score_map,img_info,h, w, sigma_h, sigma_w):
    kernal = gaussian_kernel(h, w, sigma_h, sigma_w)
    y, x = np.unravel_index(np.argmax(score_map), [len(score_map), len(score_map[0])])
    score_map[max(y-h//2,0):min(y+h//2,img_info["img_height"]), max(x-w//2,0):min(x+w//2,img_info["img_width"])] \
        = kernal[max(h//2-y,0):,max(w//2-x,0):]
    # cv2.imshow('after',score_map)
    # cv2.waitKey()
    score_map = cv2.resize(score_map, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('after',score_map)
    # cv2.waitKey()
    return score_map

def gen_center_kernel(center_map,img_info,h, w, sigma_h, sigma_w):
    kernal = gaussian_kernel(h, w, sigma_h, sigma_w)
    y, x = np.unravel_index(np.argmax(center_map), [len(center_map), len(center_map[0])])
    center_map[max(y-h//2,0):min(y+h//2,img_info["img_height"]), max(x-w//2,0):min(x+w//2,img_info["img_width"])] \
        = kernal[max(h//2-y,0):,max(w//2-x,0):]
    center_map = cv2.resize(center_map, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
    return center_map

flip_map = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10]
def create_train_data():
    print('Creating training original images...')
    print('-'*30)
    
    i = 0
    path_depth = os.path.join(image_path, 'depth_vis')
    path_ir = os.path.join(image_path, 'ir')
    train_depth = os.listdir(path_depth)

    total_imgs = len(train_depth)*2
    depth_imgs = np.ndarray((total_imgs, image_rows, image_cols, 3), dtype=np.uint8)
    ir_imgs = np.ndarray((total_imgs, image_rows, image_cols, 3), dtype=np.uint8)
    centers = np.ndarray((total_imgs, 2), dtype=np.int16)
    annotations = np.ndarray((total_imgs, 14, 2), dtype=np.int16)
    for img_info in dict:
        if(img_info["patient"] != "7"):
            depth_img = cv2.imread(os.path.join(path_depth, img_info["image_name"]),cv2.IMREAD_UNCHANGED)
            ir_img = cv2.imread(os.path.join(path_ir, img_info["image_name"]))
            depth_img_resized = cv2.resize(depth_img, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
            ir_img_resized = cv2.resize(ir_img, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
            # ir_img_resized_small = cv2.resize(ir_img, (image_cols_map, image_rows_map), interpolation=cv2.INTER_NEAREST)
            depth_imgs[i,:,:,0] = depth_img_resized
            depth_imgs[i,:,:,1] = depth_img_resized
            depth_imgs[i,:,:,2] = depth_img_resized
            depth_imgs[i+1,:,:,0] = cv2.flip(depth_img_resized,1)
            depth_imgs[i+1,:,:,1] = cv2.flip(depth_img_resized,1)
            depth_imgs[i+1,:,:,2] = cv2.flip(depth_img_resized,1)
            ir_imgs[i] = ir_img_resized
            ir_imgs[i+1] = cv2.flip(ir_img_resized, 1)
            center_map = np.zeros((int(img_info["img_height"]), int(img_info["img_width"])))
            center_map[img_info["objpos"][0]][img_info["objpos"][1]] = 1
            center_map_resized = cv2.resize(center_map, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
            center_map_resized_fliped = cv2.flip(center_map_resized, 1)
            centers[i] = np.unravel_index(np.argmax(center_map_resized), [center_map_resized.shape[0], center_map_resized.shape[1]])
            centers[i+1] = np.unravel_index(np.argmax(center_map_resized_fliped), [center_map_resized_fliped.shape[0], center_map_resized_fliped.shape[1]])
            for x in range(0,14):
                score_map = np.zeros((int(img_info["img_height"]), int(img_info["img_width"])))
                score_map[img_info["joints"][x][0]][img_info["joints"][x][1]] = 1
                score_map_resized = cv2.resize(score_map, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
                score_map_resized_fliped = cv2.flip(score_map_resized, 1)
                annotations[i][x] = np.unravel_index(np.argmax(score_map_resized), [score_map_resized.shape[0], score_map_resized.shape[1]])
                annotations[i + 1][flip_map[x]] = np.unravel_index(np.argmax(score_map_resized_fliped), [score_map_resized_fliped.shape[0], score_map_resized_fliped.shape[1]])
            # for x in range(0,14):
            #     score_map = np.zeros((image_rows, image_cols))
            #     score_map[annotations[i][x][0]][annotations[i][x][1]] = 1
            #     score_map1 = np.zeros((image_rows, image_cols))
            #     score_map1[annotations[i+1][x][0]][annotations[i+1][x][1]] = 1
            #     cv2.imshow('show',score_map)
            #     cv2.imshow('show2', score_map1)
            #     cv2.waitKey(1000)

            if i % 100 == 0:
                print('Done: {0}/{1} train original images'.format(i, total_imgs))
            i += 2

    print('Loading done.')
    np.save('./dataset/train_autoencoder_depth.npy', depth_imgs)
    np.save('./dataset/train_autoencoder_ir.npy', ir_imgs)
    np.save('./dataset/train_autoencoder_center.npy', centers)
    np.save('./dataset/train_autoencoder_annotation.npy', annotations)
    print('Saving done.')

def create_test_data():
    print('Creating test images...')
    print('-' * 30)

    i = 0
    path_depth = os.path.join(test_path, 'depth_vis')
    path_ir = os.path.join(test_path, 'ir')
    test_depth = os.listdir(path_depth)

    total_imgs = len(test_depth) * 2
    depth_imgs = np.ndarray((total_imgs, image_rows, image_cols, 3), dtype=np.uint8)
    ir_imgs = np.ndarray((total_imgs, image_rows, image_cols, 3), dtype=np.uint8)
    centers = np.ndarray((total_imgs, 2), dtype=np.int16)
    annotations = np.ndarray((total_imgs, 14, 2), dtype=np.int16)
    for img_info in dict:
        if (img_info["patient"] == "7"):
            depth_img = cv2.imread(os.path.join(path_depth, img_info["image_name"]), cv2.IMREAD_UNCHANGED)
            ir_img = cv2.imread(os.path.join(path_ir, img_info["image_name"]))
            depth_img_resized = cv2.resize(depth_img, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
            ir_img_resized = cv2.resize(ir_img, (image_cols, image_rows), interpolation=cv2.INTER_NEAREST)
            depth_img_resized = np.asarray(depth_img_resized)
            depth_imgs[i,:,:,0] = depth_img_resized
            depth_imgs[i,:,:,1] = depth_img_resized
            depth_imgs[i,:,:,2] = depth_img_resized
            depth_imgs[i+1,:,:,0] = cv2.flip(depth_img_resized, 1)
            depth_imgs[i+1,:,:,1] = cv2.flip(depth_img_resized, 1)
            depth_imgs[i+1,:,:,2] = cv2.flip(depth_img_resized, 1)
            ir_imgs[i] = ir_img_resized
            ir_imgs[i+1] = cv2.flip(ir_img_resized, 1)
            center_map = np.zeros((int(img_info["img_height"]), int(img_info["img_width"])))
            center_map[img_info["objpos"][0]][img_info["objpos"][1]] = 1
            center_map_resized = cv2.resize(center_map, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
            center_map_resized_fliped = cv2.flip(center_map_resized, 1)
            centers[i] = np.unravel_index(np.argmax(center_map_resized),
                                          [center_map_resized.shape[0], center_map_resized.shape[1]])
            centers[i + 1] = np.unravel_index(np.argmax(center_map_resized_fliped),
                                              [center_map_resized_fliped.shape[0], center_map_resized_fliped.shape[1]])
            for x in range(0, 14):
                score_map = np.zeros((int(img_info["img_height"]), int(img_info["img_width"])))
                score_map[img_info["joints"][x][0]][img_info["joints"][x][1]] = 1
                score_map_resized = cv2.resize(score_map, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
                score_map_resized_fliped = cv2.flip(score_map_resized, 1)
                annotations[i][x] = np.unravel_index(np.argmax(score_map_resized),
                                                     [score_map_resized.shape[0], score_map_resized.shape[1]])
                annotations[i + 1][flip_map[x]] = np.unravel_index(np.argmax(score_map_resized_fliped),
                                                         [score_map_resized_fliped.shape[0], score_map_resized_fliped.shape[1]])

            if i % 100 == 0:
                print('Done: {0}/{1} train original images'.format(i, total_imgs))
            i += 2


    print(depth_imgs.shape)
    print('Loading done.')
    np.save('./dataset/test_autoencoder_depth.npy', depth_imgs)
    np.save('./dataset/test_autoencoder_ir.npy', ir_imgs)
    np.save('./dataset/test_autoencoder_center.npy', centers)
    np.save('./dataset/test_autoencoder_annotation.npy', annotations)
    print('Saving done.')

if __name__ == '__main__':
    # create_train_data()
    create_test_data()
#     check()
import numpy as np
import cv2

depth_train = np.load('./dataset/train_autoencoder_depth.npy')
facked_ir_train = np.load('./dataset/faked_train.npy')
depth_test = np.load('./dataset/test_autoencoder_depth.npy')
facked_ir_test = np.load('./dataset/faked_test.npy')

def merge(number_depth, number_faked_ir):
    print(facked_ir_train.shape)
    print(facked_ir_test.shape)
    merged_train = np.ndarray((depth_train.shape[0], depth_train.shape[1], depth_train.shape[2], 3), dtype=np.float32)
    merged_test = np.ndarray((depth_test.shape[0], depth_test.shape[1], depth_test.shape[2], 3), dtype=np.float32)

    channel_index = 0
    for i in range(len(depth_train)):
        for times in range(number_depth):
            merged_train[i, :, :, channel_index] = depth_train[i,:,:,0]
            channel_index += 1
        for times in range(number_faked_ir):
            merged_train[i, :, :, channel_index] = facked_ir_train[i,:,:,0]
            channel_index += 1
        channel_index = 0

    for i in range(len(depth_test)):
        for times in range(number_depth):
            merged_test[i, :, :, channel_index] = depth_test[i, :, :, 0]
            channel_index += 1
        for times in range(number_faked_ir):
            merged_test[i, :, :, channel_index] = facked_ir_test[i, :, :, 0]
            channel_index += 1
        channel_index = 0

    np.save('./dataset/merged_train.npy', merged_train)
    np.save('./dataset/merged_test.npy', merged_test)


    for dep in merged_train:
        cv2.imshow('merge',dep/255)
        cv2.waitKey(500)

number_depth = 0
number_faked_ir = 3 - number_depth
merge(number_depth, number_faked_ir)



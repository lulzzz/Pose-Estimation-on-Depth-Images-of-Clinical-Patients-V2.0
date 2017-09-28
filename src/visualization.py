import os
import cv2
import numpy as np

depth_image_path = './images/trial_3_autoencoder/depth'
depth_target_path = './images/trial_3_autoencoder/depth_vis'



def save_depth():
    images = os.listdir(depth_image_path)
    for image_name in images:
        depth_img = cv2.imread(os.path.join(depth_image_path, image_name), cv2.IMREAD_UNCHANGED)
        image = depth_img * (255.0 / 4000.0)
        # for x in range(len(image)):
        #     for y in range(len(image[x])):
        #         if image[x][y] < 40:
        #             image[x][y] = 200
        # image = np.array(image)
        # mean = np.mean(image)
        # image = image - mean
        # image = image * 2.2 + mean * 0.8
        cv2.imwrite(os.path.join(depth_target_path, image_name), image)


if __name__ == '__main__':
    save_depth()

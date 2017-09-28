import cv2
import os
import numpy as np


image_path = 'images/trial_2/p3+5/sit/'
# image_path = '0608.png'




if __name__ == '__main__':
    #read images    
    count = 430
    train_images = os.listdir(image_path)
    for image_name in train_images[count:]:
        if len(image_name) < 20:
            depth_img = cv2.imread(os.path.join(image_path, image_name),cv2.IMREAD_UNCHANGED)
            depth_img = depth_img.astype('float32')
            depth_img *= (350.0/2500.0)
            depth_img = depth_img.astype('int8') 
            cv2.imshow('Checking', depth_img)#[20:360,15:175]
            print(str(count)+'th  image is '+image_name)
            count += 1
            if cv2.waitKey() == ord('q'):
                continue;
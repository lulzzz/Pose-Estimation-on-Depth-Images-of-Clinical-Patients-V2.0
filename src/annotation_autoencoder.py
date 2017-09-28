import cv2
import os
import numpy as np

original_rows = 512
original_cols = 424

patient = 'patient_7'
scenario = 'scenario_1'

data_path_depth = '/media/boshen/Boshen/Data_for_Boshen/'+patient+'/'+scenario+'/frames/depth'
data_path_ir = '/media/boshen/Boshen/Data_for_Boshen/'+patient+'/'+scenario+'/frames/ir'
saved_path_depth = 'images/trial_3_autoencoder/depth/'
saved_path_ir = 'images/trial_3_autoencoder/ir/'
start_index = 1850

def compare(x, y):
    x = int(x.split('.')[0])
    y = int(y.split('.')[0])
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0
                
def annotation(depth_image, ir_image, image_name):  
    image_prefix = patient + '_' + scenario + '_' 
    image_name = image_prefix + image_name
    ir_image *= 12
    cv2.namedWindow('Labeling')
    cv2.imshow('Labeling', ir_image)
    if cv2.waitKey() == ord('q'):
        print(image_name + ' is skipped') 
    elif cv2.waitKey() == ord('w'):
        cv2.imwrite(os.path.join(saved_path_depth, image_name), depth_image)
        cv2.imwrite(os.path.join(saved_path_ir, image_name), ir_image)
        print(image_name + ' is saved') 

        
if __name__ == '__main__':
    #read images    
    count = 0
    train_images = os.listdir(data_path_depth)
    train_images.sort(compare)
    for image_name in train_images[start_index:]:
        if count%2 ==0:
            depth_img = cv2.imread(os.path.join(data_path_depth, image_name),cv2.IMREAD_UNCHANGED)
            ir_img = cv2.imread(os.path.join(data_path_ir, image_name),cv2.IMREAD_UNCHANGED)
            depth_img_transposeed = depth_img.T
            depth_img_transposeed_flipped = cv2.flip(depth_img_transposeed,0)
            ir_img_transposeed = ir_img.T
            ir_img_transposeed_flipped = cv2.flip(ir_img_transposeed,0)
            img = annotation(depth_img_transposeed_flipped, ir_img_transposeed_flipped, image_name)
        count += 1
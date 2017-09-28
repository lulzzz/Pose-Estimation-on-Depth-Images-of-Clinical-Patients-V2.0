import cv2
import os
import numpy as np

original_rows = 424
original_cols = 512
image_rows = 380
image_cols = 190

data_path = '/media/boshen/Boshen/Data_for_Boshen/patient_3/scenario_58/frames/depth'
start_index = 1900
image_prefix = 'p8s14d_'
root_path = 'test/trial_2/p8/'
image_l_path = root_path + 'left/'
image_sl_path = root_path + 'slightly_left/'
image_m_path = root_path + 'middle/'
image_sr_path = root_path + 'slightly_right/'
image_r_path = root_path + 'right/'
image_sit_path = root_path + 'sit/'
image_e_path = root_path + 'empty/'

class Annotation(object):
    def __init__(self, image, image_name):   
        self.image_name = image_prefix + image_name
        self.image = image   
        self.visualization = self.image.copy()
        self.visualization = self.visualization.astype('float32')
        self.visualization *= (300.0/2500.0)
        self.visualization = self.visualization.astype('int8')  
#         self.visualization = self.visualization.astype('int8')
#         self.visualization = cv2.convertScaleAbs(self.visualization, alpha=(255/2500))
        self.i = 0
    
#     global save_img
#     def save_img(self, image_path): 
#         if image_path != '' & self.i == 0:
#             cv2.imwrite(os.path.join(image_path, self.image_name), self.image)     
#             self.i = 1
#             
#     def on_mouse(self,event, x, y, flags, param):
#             if event == cv2.EVENT_LBUTTONDOWN:
#                 if 0<= x <80:
#                     save_img(self,image_l_path)
#                     print('Current image is treated as left')
#                 elif 80<= x <160:
#                     save_img(image_sl_path)
#                     print('Current image is treated as slightly left')
#                 elif 160<= x <240:
#                     save_img(image_m_path)
#                     print('Current image is treated as middle')
#                 elif 240<= x <320:
#                     save_img(image_sr_path)
#                     print('Current image is treated as slightly right')                   
#                 elif 320<= x <400:
#                     save_img(image_r_path)
#                     print('Current image is treated as right')                    
#                 elif 400<= x <480:
#                     save_img('')
#                     print('Current image is skipped')
#                     

                
    def label_image(self):   
        #define labeling window
        cv2.namedWindow('Labeling')
        cv2.imshow('Labeling', self.visualization)
        if cv2.waitKey() == ord('q'):
            print(image_name + ' is skipped') 
        elif cv2.waitKey() == ord('1'):
            cv2.imwrite(os.path.join(image_l_path, self.image_name), self.image)
            cv2.imwrite(os.path.join(image_r_path, self.image_name), cv2.flip(self.image,1))
            print(image_name + ' is treated as left') 
        elif cv2.waitKey() == ord('2'):
            cv2.imwrite(os.path.join(image_sl_path, self.image_name), self.image)
            cv2.imwrite(os.path.join(image_sr_path, self.image_name), cv2.flip(self.image,1))
            print(image_name + ' is treated as slightly left') 
        elif cv2.waitKey() == ord('3'):
            cv2.imwrite(os.path.join(image_m_path, self.image_name), self.image)
            print(image_name + ' is treated as middle') 
        elif cv2.waitKey() == ord('4'):
            cv2.imwrite(os.path.join(image_sr_path, self.image_name), self.image)
            cv2.imwrite(os.path.join(image_sl_path, self.image_name), cv2.flip(self.image,1))
            print(image_name + ' is treated as slightly right') 
        elif cv2.waitKey() == ord('5'):
            cv2.imwrite(os.path.join(image_r_path, self.image_name), self.image)
            cv2.imwrite(os.path.join(image_l_path, self.image_name), cv2.flip(self.image,1))
            print(image_name + ' is treated as right') 
        elif cv2.waitKey() == ord('6'):
            cv2.imwrite(os.path.join(image_sit_path, self.image_name), self.image)
#             image_flipped_name = self.image_name.split('.')[0] + '_flipped.png'
#             cv2.imwrite(os.path.join(image_sit_path, image_flipped_name), cv2.flip(self.image,1))
            print(image_name + ' is treated as sit') 
        elif cv2.waitKey() == ord('7'):
            cv2.imwrite(os.path.join(image_e_path, self.image_name), self.image)
            image_flipped_name = self.image_name.split('.')[0] + '_flipped.png'
            cv2.imwrite(os.path.join(image_e_path, image_flipped_name), cv2.flip(self.image,1))
            print(image_name + ' is treated as empty') 
        
if __name__ == '__main__':
    #read images    
    count = 0
    train_images = os.listdir(data_path)
    for image_name in train_images[start_index:]:
        if count%2 ==0:
            depth_img = cv2.imread(os.path.join(data_path, image_name),cv2.IMREAD_UNCHANGED)
    #         M_rotate  = cv2.getRotationMatrix2D((original_cols/2,original_rows/2),90,1)
    #         img_rotated = cv2.warpAffine(depth_img, M_rotate, (original_cols, original_rows))
    #         img_rotated_croped = img_rotated[0:380,100:290].copy()
    
            img_transposeed = depth_img.T
            img_transposeed_flipped = cv2.flip(img_transposeed,0)
            result = img_transposeed_flipped[30:480,110:300]
            img_rotated_croped_resized = cv2.resize(result, (image_cols, image_rows),interpolation=cv2.INTER_NEAREST)
            img = Annotation(img_rotated_croped_resized, image_name)
            img.label_image()
        count += 1
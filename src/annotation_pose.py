import cv2
import os
import numpy as np

image_rows = 512
image_cols = 512
data_path = 'D:/patient_3/scenario_20/frames/depth/'
start_index = 1000
image_prefix = 'p3s9d_'
image_path = 'images/'
label_path = 'annotations/'

joints = {1:['head',(33,144,255)], 2:['neck',(240,248,255)],
          3:['lShoulder',(123,104,238)], 4:['rShoulder',(106,90,205)],
          5:['lElbow',(255,20,147)], 6:['rElbow',(199,21,133)],
          7:['lHand',(0,255,255)], 8:['rHand',(0,206,209)]}

class Annotation(object):
    def __init__(self, image, image_name):   
        self.image_name = image_prefix + image_name
        self.image_mask_name = image_prefix + 'mask_'+ image_name
        self.image = image
        self.protected_image = self.image.copy()
        self.image = self.image.astype('float32')
        self.image *= (255.0/2500.0)
        self.image = self.image.astype('int8')  
        #new an image for annotation
        self.white_board = np.zeros((image_rows, image_cols), dtype=np.uint8)
        self.imgs_mask = self.white_board.copy()
        self.imgs_mask_demo = self.white_board.copy()
        self.i = 0
        
    def on_mouse(self,event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.i += 1
                if self.i <= 8:
                    cv2.circle(self.image,(x,y),4,200,-1)
                    cv2.circle(self.imgs_mask,(x,y),4,(self.i),-1)
                    cv2.circle(self.imgs_mask_demo,(x,y),4,(self.i*20),-1)
                    print(str(self.i) + '/' + str(len(joints)) + '--' + joints[self.i][0] + '({}, {})'.format(x, y))
                    if self.i == 8:
                        print('Current image is labeled, click anywhere for saving.')
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.i += 1
                print(str(self.i) + '/' + str(len(joints)) + '-- skip: ' + joints[self.i][0])
                
                
                
    def label_image(self):   
        #define labeling window
        cv2.namedWindow('Labeling')
        cv2.setMouseCallback('Labeling', self.on_mouse)
        
        while(1):
            cv2.imshow('Labeling', self.image)
#             cv2.imshow('Labeling', cv2.convertScaleAbs(self.image, alpha=(255.0/2500.0)))
            cv2.imshow('Mask', self.imgs_mask_demo)
            if self.i == 9: 
                #save original image
                cv2.imwrite(os.path.join(image_path, self.image_name), self.protected_image)
                cv2.imwrite(os.path.join(label_path, self.image_mask_name), self.imgs_mask)
                print('Both depth image and mask are saved.')
                cv2.waitKey(500)    
                break
            
            if cv2.waitKey(33) == ord('q'):
                print('Skip:'+self.image_name)
                break
            
            if cv2.waitKey(33) == ord('c'):       
                print('All labels have been cleaned from current image, please relabel.')   
                self.i = 0     
                #reset current images
                self.image = self.protected_image.copy()
                self.image = self.image.astype('float32')
                self.image *= (255.0/2500.0)
                self.image = self.image.astype('int8')
                
                self.imgs_mask = self.white_board.copy()
                self.imgs_mask_demo = self.white_board.copy()
        
        
if __name__ == '__main__':
    #read images    
    train_images = os.listdir(data_path)
    for image_name in train_images[start_index:]:
        depth_img = cv2.imread(os.path.join(data_path, image_name),cv2.IMREAD_UNCHANGED)
        img_resized = cv2.resize(depth_img, (image_cols, image_rows),interpolation=cv2.INTER_NEAREST)
        M_rotate  = cv2.getRotationMatrix2D((image_cols/2,image_rows/2),90,1)
        img_resized_rotated = cv2.warpAffine(img_resized, M_rotate, (image_cols, image_rows))
        img = Annotation(img_resized_rotated, image_name)
        img.label_image()
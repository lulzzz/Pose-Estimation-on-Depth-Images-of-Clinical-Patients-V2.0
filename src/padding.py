
import cv2
import os
import numpy as np


image_path = 'visualization/depth/'
target_path = 'visualization/ts_depth/'


def padding(images,PH = 376, PW =656):
    i = 0
    padded_images = np.ndarray((len(images),  PH, PW, 3), dtype=np.float32)
    for image in images:
        image = image[0]
        padding = int((len(image)*1.74-len(image[0]))//2)
        img_two = np.ones((len(image),int(len(image)*1.74),3))
        img_two[:,padding:len(image[0])+padding,0]=image
        img_two[:,padding:len(image[0])+padding,1]=image
        img_two[:,padding:len(image[0])+padding,2]=image
        img_two = cv2.resize(img_two, (PW, PH),interpolation=cv2.INTER_CUBIC)
        padded_images[i] = img_two
        i += 1
    # np.save('./result/autoencoder.npy', padded_images)
    return  padded_images


def padding_single(image,PH = 376, PW =656):
    padding = int((len(image) * 1.74 - len(image[0])) // 2)
    img_two = np.ones((len(image),int(len(image)*1.74),3))
    img_two[:,padding:len(image[0])+padding,0]=image
    img_two[:,padding:len(image[0])+padding,1]=image
    img_two[:,padding:len(image[0])+padding,2]=image
    img_two = cv2.resize(img_two, (PW, PH),interpolation=cv2.INTER_CUBIC)

    # np.save('./result/autoencoder.npy', padded_images)
    return  img_two

if __name__ == '__main__':
    test_predict = np.load('./result/autoencoder.npy')
    index = 0
    test_predict = padding(test_predict)
    for image in test_predict:
        image = image*255
        # image = image.astype('int8')
        cv2.imwrite(os.path.join('./result/autoencoder', str(index)+'.png'),image)
        index += 1

    # images = os.listdir('visualization/depth/')
    # for image_name in images:
    #     image = cv2.imread(os.path.join(image_path, image_name),cv2.IMREAD_UNCHANGED)
    #     cv2.imwrite(os.path.join('visualization/ts_depth/', image_name), padding_single(image))
                    
                    
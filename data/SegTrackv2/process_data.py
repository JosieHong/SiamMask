import os
import glob

import numpy as np

import cv2
from PIL import Image

def add_mask(mask, new_mask, num):
    width = mask.size[0]
    height = mask.size[1]

    for i in range(height):
        for j in range(width):
            if mask.getpixel((j,i)) != (0,0,0):
                new_mask.putpixel((j,i), num)
    new_mask.putpalette([0,0,0, 255,0,0, 0,255,0, 0,0,255, 255,255,0, 255,0,255, 0,255,255])

    return new_mask

def main():
    videos = open('./ImageSets/all.txt', 'r').readlines()
    for video in videos:
        # check output path
        if not os.path.exists(os.path.join('./Annotations', video[1:-1])):
            os.makedirs(os.path.join('./Annotations', video[1:-1]))
            print("mkdir {}".format(os.path.join('./Annotations', video[1:-1])))

        # convert *.bmp to *.png
        image_names = sorted(glob.glob(os.path.join('./JPEGImages', video[1:-1], '*.png')))
        if len(image_names) == 0:
            temps = sorted(glob.glob(os.path.join('./JPEGImages', video[1:-1], '*.bmp')))
            for temp in temps:
                temp_image = Image.open(temp)
                temp_image.save(temp[:-4]+'.png')
                print("save {}".format(temp[:-4]+'.png'))
                os.remove(temp)
                print("remove {}".format(temp))

        # count objects number
        object_num = 0
        files = sorted(os.listdir(os.path.join('./GroundTruth', video[1:-1])))
        for file_name in files:
            if os.path.isdir(os.path.join('./GroundTruth', video[1:-1], file_name)):
                object_num +=1
        if object_num == 0:
            object_num += 1
        print('There are {} objects in {}.'.format(object_num, video[1:-1]))

        # get mask names
        mask_names = []
        if object_num == 1: 
            mask_names.append(sorted(glob.glob(os.path.join('./GroundTruth', video[1:-1], '*.png'))))
            # for *.bmp
            if len(mask_names[0]) == 0:
                mask_names[0] = sorted(glob.glob(os.path.join('./GroundTruth', video[1:-1], '*.bmp')))
        else:
            for i in range(object_num):
                mask_names.append(sorted(glob.glob(os.path.join('./GroundTruth', video[1:-1], str(i+1), '*.png'))))
                # for *.bmp
                if len(mask_names[i]) == 0:
                    mask_names[i] = sorted(glob.glob(os.path.join('./GroundTruth', video[1:-1], str(i+1), '*.bmp')))
        # josie.debug
        # print(mask_names)

        # add mask
        frame_num = len(mask_names[0])
        print("frame number: {}".format(frame_num))
        for j in range(frame_num):
            for i in range(object_num):
                mask = Image.open(mask_names[i][j])
                if i == 0:
                    new_mask = Image.new('P', (mask.size[0], mask.size[1]), 0)
                    print("new mask")
                new_mask = add_mask(mask, new_mask, i+1)
                if i == object_num-1:
                    new_mask.save('./Annotations/{}/{}.png'.format(video[1:-1], str(j).zfill(5)))
                    print("save ./Annotations/{}/{}.png".format(video[1:-1], str(j).zfill(5)))

        '''
        print(image_names)
        for image_name in image_names:
            mask = cv2.imread(image_name)
            new_mask = trans_mask(mask)
            cv2.imwrite('./Annotations{}'.format(image_name[13:]), new_mask)
            # convert image mode
            image = Image.open(image_name)
            image = image.convert('P')
            image.save('./Annotations{}.png'.format(image_name[13:-4]))
            print('save ./Annotations{}.png'.format(image_name[13:-4]))
        '''

if __name__ == '__main__':
    main()
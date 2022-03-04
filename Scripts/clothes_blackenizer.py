import numpy as np
import glob
import shutil
import os
import cv2
from PIL import Image, ImageOps
from matplotlib import pyplot as plt


clothes_dir = './cloth'
clothes_mask_dir = './cloth-mask'
image_dir = './image'
image_parse_dir = './image-parse'
result_dir = './result'


def load_one_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # img.save('img_test.jpg', format='jpeg')
    np_img = np.array(img)

    return np_img

def load_one_image_parse(image_parse_path):
    # img_parse = Image.open(image_parse_path).convert('RGB')
    img_parse = Image.open(image_parse_path)
    # img_parse.save('img_parse_test.png', format='png')
    np_img_parse = np.array(img_parse)

    return np_img_parse

def get_parse_clothes(img_parse):
    """
    img_parse: numpy array
    """
    print(np.unique(img_parse))
    parse_upper = ((img_parse == 5).astype(np.float32) +
                   (img_parse == 6).astype(np.float32) +
                   (img_parse == 7).astype(np.float32))

    print("parse_cloth's elements:", np.unique(parse_upper))

    return parse_upper

def parse2mask(parse):
    """
    parse: NUMPY ARRAY upper clothes
    """
    upper_mask = parse[np.where(parse > 0.0)] = 1.0



def clothes_darkenizer(img, mask):
    print("mask", mask.shape)
    np_clothes = np.copy(img)
    print(type(np_clothes), np_clothes.shape)
    np_clothes[np.where(mask == 0.0)] = 0.0 # only clothes will survive


    Image.fromarray(np.uint8(np_clothes)).save('np_clothes.jpg')
    PIL_clothes = Image.fromarray(np.uint8(np_clothes)).convert('RGB')
    PIL_clothes.save('PIL_clothes.jpg')
    PIL_gray_clothes = ImageOps.grayscale(PIL_clothes)

    PIL_gray_clothes.save('gray_PIL.jpg')

    np_gray_clothes = np.array(PIL_gray_clothes)

    # stack three times
    np_gray_clothes = np.stack([np_gray_clothes,np_gray_clothes,np_gray_clothes], axis=-1)

    return np_gray_clothes

def merge_images(img1, img2, img2_mask):
    """
    img1: main image
    img2: sub image
    img2_mask
    """
    result = np.copy(img1)
    result[np.where(img2_mask != 0)] = img2[np.where(img2_mask != 0)]

    return result



def main():
    shutil.rmtree(result_dir) if os.path.exists(result_dir) else None
    os.mkdir(result_dir) if not os.path.exists(result_dir) else None

    result_cloth_dir = os.path.join(result_dir, 'cloth')
    result_cloth_mask_dir = os.path.join(result_dir, 'cloth-mask')
    result_image_dir = os.path.join(result_dir, 'image')
    result_image_parse_dir = os.path.join(result_dir, 'image-parse')

    os.mkdir(result_cloth_dir)
    os.mkdir(result_cloth_mask_dir)
    os.mkdir(result_image_dir)
    os.mkdir(result_image_parse_dir)

    # human image processing
    for img_path in glob.glob(os.path.join(image_dir, '*.jpg')):
        img_parse_path = os.path.join(image_parse_dir, os.path.basename(img_path)).replace('.jpg', '.png')

        img = load_one_image(img_path)
        img_parse = load_one_image_parse(img_parse_path)

        parse_upper = get_parse_clothes(img_parse)
        np_gray_clothes = clothes_darkenizer(img, parse_upper)

        result_img = merge_images(img, np_gray_clothes, parse_upper)
        PIL_result_img = Image.fromarray(result_img)
        PIL_result_img.save(os.path.join(result_image_dir, os.path.basename(img_path)))
        Image.fromarray(img_parse).save(os.path.join(result_image_parse_dir, os.path.basename(img_parse_path)))

        plt.imshow(np.array(result_img))
        plt.show()

    # clothes image processing
    for clothes_path in glob.glob(os.path.join(clothes_dir, '*.jpg')):
        clothes_mask_path = os.path.join(clothes_mask_dir, os.path.basename(clothes_path))

        clothes = load_one_image(clothes_path)
        clothes_mask = load_one_image(clothes_mask_path)

        np_gray_clothes = clothes_darkenizer(clothes, clothes_mask)

        result_img = merge_images(clothes, np_gray_clothes, clothes_mask)
        PIL_result_img = Image.fromarray(result_img)
        PIL_result_img.save(os.path.join(result_cloth_dir, os.path.basename(clothes_path)))
        Image.fromarray(clothes_mask).save(os.path.join(result_cloth_mask_dir, os.path.basename(clothes_mask_path)))

        plt.imshow(np.array(result_img))
        plt.show()

if __name__ == '__main__':
    main()
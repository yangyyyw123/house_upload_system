import cv2
import numpy as np
import os
import math

def pad_to_multiple(img, block_size):
    h, w = img.shape[:2]
    new_h = math.ceil(h / block_size) * block_size
    new_w = math.ceil(w / block_size) * block_size

    top = (new_h - h) // 2
    bottom = new_h - h - top
    left = (new_w - w) // 2
    right = new_w - w - left

    color = [255] * img.shape[2] if len(img.shape) == 3 else 255
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_img

def batch_split_images(folder_path, save_path, block_size=512):
    """
    批量处理文件夹中的图片，若尺寸不是block_size的整数倍，先白色填充，
    再分割成block_size大小保存到对应子文件夹。

    参数:
        folder_path (str): 待处理图片文件夹路径
        block_size (int): 切割尺寸，默认119

    返回:
        int: 成功处理的图片数量
    """
    file_list = os.listdir(folder_path)
    num_processed = 0

    for file_name in file_list:
        pic_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(pic_path):
            continue  # 跳过非文件

        pic_target = os.path.join(save_path, f"divide_{os.path.splitext(file_name)[0]}")
        if not os.path.exists(pic_target):
            os.makedirs(pic_target)

        picture = cv2.imread(pic_path)
        if picture is None:
            print(f"警告：无法读取图片 {pic_path}，跳过。")
            continue

        height, width = picture.shape[:2]
        print(f"处理图片: {file_name}，尺寸: {height}x{width}")

        # 判断是否需要填充
        if height % block_size == 0 and width % block_size == 0:
            img_to_cut = picture
            print("尺寸满足要求，无需填充。")
        else:
            img_to_cut = pad_to_multiple(picture, block_size)
            print(f"尺寸不满足，已填充至 {img_to_cut.shape[0]}x{img_to_cut.shape[1]}")

        num_height = img_to_cut.shape[0] // block_size
        num_width = img_to_cut.shape[1] // block_size

        for i in range(num_height):
            for j in range(num_width):
                patch = img_to_cut[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                save_img_path = os.path.join(pic_target, f"{i+1}_{j+1}.jpg")
                cv2.imwrite(save_img_path, patch)

        print(f"{file_name} 切割完成，保存至 {pic_target}")
        num_processed += 1

    print(f"总共处理图片数量: {num_processed}")
    return num_processed

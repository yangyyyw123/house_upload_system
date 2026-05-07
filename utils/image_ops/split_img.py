import os
import cv2
import math

def padding(img, block_size):
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

def split_and_pad_img(pic_path, save_folder, block_size=512):
    """
    判断图像尺寸是否为block_size的整数倍，
    不是则先填充到整数倍，再切割保存。
    返回切割的小图数量。
    """
    filename = os.path.basename(pic_path)
    pic_target = os.path.join(save_folder, f"divide_{os.path.splitext(filename)[0]}")
    if not os.path.exists(pic_target):
        os.makedirs(pic_target)

    image = cv2.imread(pic_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {pic_path}")

    height, width = image.shape[:2]

    # 判断是否是整数倍
    if height % block_size == 0 and width % block_size == 0:
        img_to_cut = image
        print("图像尺寸符合要求，无需填充。")
    else:
        print("图像尺寸不符合要求，进行填充。")
        img_to_cut = padding(image,block_size)

    h_new, w_new = img_to_cut.shape[:2]
    num_height = h_new // block_size
    num_width = w_new // block_size

    count = 0
    for i in range(num_height):
        for j in range(num_width):
            patch = img_to_cut[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            save_path = os.path.join(pic_target, f"{i+1}_{j+1}.jpg")
            cv2.imwrite(save_path, patch)
            count += 1

    print(f"切割后图像尺寸: {img_to_cut.shape}")
    print(f"共切割成 {count} 张图，保存在: {save_folder}")

    return count

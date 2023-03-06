import numpy as np
import torch
import cv2
from main_func.main_components.utils.image import show_cam_on_image


def origin_creator(img:torch.Tensor, organ_groups:int=1):
    B, C, L, W = img.shape  # # [batch, 3, y, x]
    if C == 3:
        array_img = img.data.numpy()  # 保存numpy版本用于他用d
        min_value = np.min(array_img)
        array_img = (array_img - min_value)/(np.max(array_img) - min_value + 1e-10)
        
        img_color_group = np.zeros([B, organ_groups, C, L, W])
        for batches in range(B):
            for slice in range(organ_groups):
                img_color_group[batches, slice, :] = array_img[batches, :]
        img_color_group = img_color_group.transpose(0, 1, 3, 4, 2)  # [batch, organ_groups, y, x, 3] 用于后面画图 -- CAM覆盖使用的原图
    elif C == 1:
        array_img = img.data.numpy()  # 保存numpy版本用于他用d
        img_color_group = np.zeros([B, organ_groups, 3, L, W])
        for batches in range(B):
            for slice in range(organ_groups):
                for channel in range(3):
                    img_color_group[batches, slice, channel, :] = array_img[batches, :]
        img_color_group = img_color_group.transpose(0, 1, 3, 4, 2)
    elif C % 5 == 0:
        array_img = img.data.numpy()  # [B, C, L, W]
        img_color_group = np.zeros([B, organ_groups, 3, L, W])
        for batches in range(B):
            for slice in range(organ_groups):  # [C, L, W]
                for channel in range(3):
                    img_color_group[batches, slice, channel, :] = array_img[batches, (slice*5+2),:]
        img_color_group = img_color_group.transpose(0, 1, 3, 4, 2)        
    else:
        raise ValueError(f'Not valid task, channel number: {C}')
    return img_color_group


def cam_creator(grayscale_cam, predict_category, confidence, organ_groups, origin_img, use_origin:bool=True):
    single_grayscale_cam, single_predict_category, single_confidence = grayscale_cam, predict_category, confidence
    cam_image_group_list = []
    for j in range(organ_groups):
        grayscale_cam_group = single_grayscale_cam[j, :]
        cam_image_group = show_cam_on_image(origin_img[j], grayscale_cam_group, use_rgb=True, use_origin=use_origin)
        cam_image_group = cv2.cvtColor(cam_image_group, cv2.COLOR_RGB2BGR)
        cam_image_group_list.append(cam_image_group)
    
    output_label = single_predict_category
    cf_num = str(np.around(single_confidence, decimals=3))
    origin_list = []
    for img in origin_img:  # img_color_group [batch, organ_groups, y, x, 3] 
        origin_list.append(img * 255)  # img - [L, W]

    concat_img_origin = cv2.hconcat(origin_list) # [x, N*y, 3]

    concat_img_cam = cv2.hconcat(cam_image_group_list) # [3, x, N*y]
    concat_img_cam = concat_img_cam.astype(concat_img_origin.dtype)
    concat_img_all = cv2.vconcat([concat_img_cam, concat_img_origin])
    return concat_img_all, output_label, cf_num

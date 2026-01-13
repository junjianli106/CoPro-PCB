import cv2
import os
import numpy as np
from tqdm import tqdm

def specify_resolution(image_list, score_list, mask_list, resolution: tuple=(400,400)):
    """
    批量处理图像resize，提高效率
    """
    # 如果列表为空，直接返回
    if len(image_list) == 0:
        return [], [], []
    
    # 批量处理：将列表转换为numpy数组进行批量resize
    resize_image = []
    resize_score = []
    resize_mask = []
    
    # 批量处理resize（每次处理一批，避免内存问题）
    batch_size = 32  # 可以根据内存情况调整
    total = len(image_list)
    
    for i in tqdm(range(0, total, batch_size), desc="Resizing images"):
        batch_end = min(i + batch_size, total)
        batch_images = image_list[i:batch_end]
        batch_scores = score_list[i:batch_end]
        batch_masks = mask_list[i:batch_end]
        
        # 批量resize
        for image, score, mask in zip(batch_images, batch_scores, batch_masks):
            # 确保输入是numpy数组
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            if not isinstance(score, np.ndarray):
                score = np.array(score)
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            
            # Resize操作
            resized_image = cv2.resize(image, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
            resized_score = cv2.resize(score, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
            resized_mask = cv2.resize(mask, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)
            
            # 确保resize后的mask是二值的（0或1），因为resize可能产生中间值
            # 检查mask的值范围，如果已经是0或1，使用>0；如果是0-255，使用>127
            if resized_mask.dtype != np.uint8:
                resized_mask = resized_mask.astype(np.uint8)
            # 检查mask的最大值，如果<=1，说明已经是二值化的，使用>0；否则使用>127
            if resized_mask.max() <= 1:
                resized_mask = (resized_mask > 0).astype(np.uint8)
            else:
                resized_mask = (resized_mask > 127).astype(np.uint8)
            
            resize_image.append(resized_image)
            resize_score.append(resized_score)
            resize_mask.append(resized_mask)

    return resize_image, resize_score, resize_mask

def normalize(scores):

    max_value = np.max(scores)
    min_value = np.min(scores)

    norml_scores = (scores - min_value) / (max_value - min_value)
    return norml_scores

def save_single_result(classification_score, segmentation_score, root_dir, shot_name, experiment_indx, subset_name, defect_type, name, use_defect_type):

    if use_defect_type:
        # mvtec2d mvtec3d
        save_dir = os.path.join(root_dir, shot_name, experiment_indx, subset_name, defect_type)
    else:
        # visa
        save_dir = os.path.join(root_dir, shot_name, experiment_indx, subset_name)

    os.makedirs(save_dir, exist_ok=True)

    classification_dir = os.path.join(save_dir, 'classification')
    segmentation_dir = os.path.join(save_dir, 'segmentation')
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)

    classification_path = os.path.join(classification_dir, f'{name}.txt')
    segmentation_path = os.path.join(segmentation_dir, f'{name}.npz')

    with open(classification_path, "w") as f:
        f.write(f'{classification_score:.5f}')

    segmentation_score = np.round(segmentation_score * 255).astype(np.uint8)
    np.savez_compressed(segmentation_path, img=segmentation_score)

def save_results(classification_score_list, segmentation_score_list, root_dir, shot_name, experiment_indx, name_list, use_defect_type):

    for classification_score, segmentation_score, full_name in zip(classification_score_list,
                                                                           segmentation_score_list,
                                                                           name_list):
        subset_name, defect_type, name = full_name.split('-')
        save_single_result(classification_score, segmentation_score, root_dir, shot_name, experiment_indx, subset_name, defect_type, name, use_defect_type)

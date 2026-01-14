import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image

class CLIPDataset(Dataset):
    def __init__(self, load_function, category, phase, k_shot, seed, transform=None, img_resize=None, img_cropsize=None):

        self.load_function = load_function
        self.phase = phase
        self.category = category
        self.transform = transform
        self.img_resize = img_resize
        self.img_cropsize = img_cropsize

        # load datasets
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(k_shot, seed)  # self.labels => good : 0, anomaly : 1

        

    def load_dataset(self, k_shot, seed):

        (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
        (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types) = self.load_function(self.category,
                                                                                                      k_shot,
                                                                                                      seed)
        if self.phase == 'train':

            return train_img_tot_paths, \
                   train_gt_tot_paths, \
                   train_tot_labels, \
                   train_tot_types
        else:
            return test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if gt == 0:
            gt = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        else:
            gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                # 如果GT文件读取失败，创建全零mask
                print(f'Warning: Failed to load GT mask: {gt}, using zero mask')
                gt = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
            else:
                # 检查原始GT mask的值分布（仅在前几个样本中打印，避免输出过多）
                if idx < 3:  # 只打印前3个样本的调试信息
                    unique_vals = np.unique(gt)
                    print(f'[DEBUG] Sample {idx}: GT mask unique values (before processing): {unique_vals[:20]}... (showing first 20)')
                    print(f'[DEBUG] Sample {idx}: GT mask value range: [{gt.min()}, {gt.max()}]')
                    print(f'[DEBUG] Sample {idx}: GT mask shape: {gt.shape}')
                
                # 使用更严格的阈值：通常GT mask应该是二值的（0或255），但有些可能是0-255的灰度图
                # 使用阈值127来区分背景和前景，更安全
                # 但为了兼容性，我们仍然使用>0，因为标准格式应该是0和255
                # 如果原始GT mask有很多中间值，可能需要调整阈值
                gt_binary = (gt > 127).astype(np.uint8) * 255  # 使用127作为阈值，更严格
                gt = gt_binary
                
                # 确保GT mask和图像尺寸一致
                if gt.shape[:2] != img.shape[:2]:
                    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # resize后重新二值化，因为resize可能产生中间值
                    gt = (gt > 127).astype(np.uint8) * 255

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'

        if self.transform is not None:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = self.transform(img_pil)

            from PIL import Image as PILImage
            gt_pil = PILImage.fromarray(gt, mode='L')
            if self.img_resize is not None and self.img_cropsize is not None:
                # 应用Resize
                gt_pil = gt_pil.resize((self.img_resize, self.img_resize), PILImage.NEAREST)
                # 应用CenterCrop
                width, height = gt_pil.size
                left = (width - self.img_cropsize) // 2
                top = (height - self.img_cropsize) // 2
                right = left + self.img_cropsize
                bottom = top + self.img_cropsize
                gt_pil = gt_pil.crop((left, top, right, bottom))
                # 转换回numpy array
                gt = np.array(gt_pil, dtype=np.uint8)
                gt = (gt > 127).astype(np.uint8) * 255

        return img, gt, label, img_name, img_type


# PCB区域自动检测
def detect_pcb_region(img, padding=10):
    """自动检测PCB主要区域"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 形态学操作填充小孔洞
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 查找最大轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # 如果没有检测到轮廓，返回整个图像区域
        return (0, 0, img.shape[1], img.shape[0])

    max_contour = max(contours, key=cv2.contourArea)

    # 获取最小外接矩形
    x, y, w, h = cv2.boundingRect(max_contour)

    # 确保 padding 不会超出图像范围
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(w + 2 * padding, img.shape[1] - x)
    h = min(h + 2 * padding, img.shape[0] - y)

    return (x, y, w, h)


# 在PCB区域内生成随机小区域
def get_random_subroi(pcb_region, min_size=30, max_size=80):
    """在PCB区域内生成随机子区域"""
    x_pcb, y_pcb, w_pcb, h_pcb = pcb_region

    # 如果 PCB 区域无效，返回整个区域
    if w_pcb <= 0 or h_pcb <= 0:
        return (x_pcb, y_pcb, w_pcb, h_pcb)

    # 确保子区域的最小尺寸不超过PCB区域
    min_size = min(min_size, w_pcb, h_pcb)
    max_size = min(max_size, w_pcb, h_pcb)

    # 如果 min_size 或 max_size 无效，返回整个区域
    if min_size <= 0 or max_size <= 0:
        return (x_pcb, y_pcb, w_pcb, h_pcb)

    sub_w = random.randint(min_size, max_size)
    sub_h = random.randint(min_size, max_size)

    # 确保子区域不会超出PCB区域
    x = x_pcb + random.randint(0, max(0, w_pcb - sub_w))
    y = y_pcb + random.randint(0, max(0, h_pcb - sub_h))

    return (x, y, sub_w, sub_h)


def pcb_elastic_deformation(img, sub_roi, intensity=1000, blur_kernel_size=25, sigma=10):
    """仅在指定子区域进行弹性形变"""
    h, w = img.shape[:2]
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    # 在子区域生成随机位移
    x, y, sub_w, sub_h = sub_roi

    # 检查子区域是否超出图像范围
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    sub_w = min(sub_w, w - x)
    sub_h = min(sub_h, h - y)

    if sub_w <= 0 or sub_h <= 0:
        return img  # 如果子区域无效，直接返回原图

    sub_dx = np.random.rand(sub_h, sub_w) * 2 - 1  # [-1,1]
    sub_dy = np.random.rand(sub_h, sub_w) * 2 - 1

    # 应用高斯模糊使边缘过渡自然
    sub_dx = cv2.GaussianBlur(sub_dx, (blur_kernel_size, blur_kernel_size), sigma) * intensity
    sub_dy = cv2.GaussianBlur(sub_dy, (blur_kernel_size, blur_kernel_size), sigma) * intensity

    dx[y:y + sub_h, x:x + sub_w] = sub_dx
    dy[y:y + sub_h, x:x + sub_w] = sub_dy

    # 应用变形
    map_x = np.float32(np.tile(np.arange(w), (h, 1)) + dx)
    map_y = np.float32(np.tile(np.arange(h), (w, 1)).T + dy)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


def pcb_scratch(img, sub_roi):
    """在子区域内生成真实划痕"""
    x, y, w, h = sub_roi
    img = img.copy()

    # 生成随机方向线段
    # 调整起始点和结束点的范围，使划痕更长
    pt1 = (x + random.randint(0, w // 4), y + random.randint(0, h // 4))  # 起始点
    pt2 = (x + 3 * w // 4 + random.randint(0, w // 4), y + 3 * h // 4 + random.randint(0, h // 4))  # 结束点

    # 划痕样式参数
    color = (40, 40, 40)  # 深灰色
    thickness = 3 + random.randint(5, 15)  # 增加划痕宽度

    # 绘制主划痕
    cv2.line(img, pt1, pt2, color, thickness)

    # 添加更多分支，使划痕更复杂
    for _ in range(random.randint(2, 5)):  # 增加分支数量
        branch_length = random.randint(10, 30)  # 增加分支长度
        angle = random.uniform(-45, 45)  # 随机角度
        branch_pt = (
            int(pt1[0] + branch_length * np.cos(np.radians(angle))),
            int(pt1[1] + branch_length * np.sin(np.radians(angle)))
        )
        cv2.line(img, pt1, branch_pt, color, max(1, thickness - 2))  # 分支稍细

    return img


class CLIPDatasetWAnom(Dataset):
    def __init__(self, load_function, category, phase, k_shot, seed, transform=None, img_resize=None, img_cropsize=None):

        self.load_function = load_function
        self.phase = phase
        self.category = category
        self.transform = transform
        self.img_resize = img_resize
        self.img_cropsize = img_cropsize

        # load datasets
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(k_shot,
                                                                                   seed)  # self.labels => good : 0, anomaly : 1

    def load_dataset(self, k_shot, seed):

        (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
            (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types) = self.load_function(self.category,
                                                                                                          k_shot,
                                                                                                          seed)
        if self.phase == 'train':

            return train_img_tot_paths, \
                train_gt_tot_paths, \
                train_tot_labels, \
                train_tot_types
        else:
            return test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        anom_img = img.copy()
        pcb_region = detect_pcb_region(anom_img)
        random_score = random.random()
        if random_score < 0.3:
            # 缺陷1：局部弹性形变
            sub_roi1 = get_random_subroi(pcb_region)
            deformed = pcb_elastic_deformation(anom_img.copy(), sub_roi1)
            anom_img = deformed
        elif random_score < 0.5:
            # 缺陷2：表面划痕
            sub_roi2 = get_random_subroi(pcb_region)
            scratched = pcb_scratch(anom_img.copy(), sub_roi2)
            anom_img = scratched
        else:
            # 缺陷1：局部弹性形变
            sub_roi1 = get_random_subroi(pcb_region)
            deformed = pcb_elastic_deformation(anom_img.copy(), sub_roi1)
            anom_img = deformed
            # 缺陷2：表面划痕
            sub_roi2 = get_random_subroi(pcb_region)
            scratched = pcb_scratch(anom_img.copy(), sub_roi2)
            anom_img = scratched

        if gt == 0:
            gt = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        else:
            gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                # 如果GT文件读取失败，创建全零mask
                print(f'Warning: Failed to load GT mask: {gt}, using zero mask')
                gt = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
            else:
                # 使用更严格的阈值：使用127作为阈值，更安全
                gt_binary = (gt > 127).astype(np.uint8) * 255
                gt = gt_binary
                # 确保GT mask和图像尺寸一致
                if gt.shape[:2] != img.shape[:2]:
                    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # resize后重新二值化
                    gt = (gt > 127).astype(np.uint8) * 255

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'

        # 如果提供了transform，应用预处理
        if self.transform is not None:
            # 转换BGR到RGB并应用transform（transform中会进行resize）
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            anom_img_pil = Image.fromarray(cv2.cvtColor(anom_img, cv2.COLOR_BGR2RGB))
            img = self.transform(img_pil)
            anom_img = self.transform(anom_img_pil)
            
            # GT mask也需要同步resize到相同的尺寸（img_cropsize x img_cropsize）
            from PIL import Image as PILImage
            gt_pil = PILImage.fromarray(gt, mode='L')
            if self.img_resize is not None and self.img_cropsize is not None:
                # 应用Resize
                gt_pil = gt_pil.resize((self.img_resize, self.img_resize), PILImage.NEAREST)
                # 应用CenterCrop
                width, height = gt_pil.size
                left = (width - self.img_cropsize) // 2
                top = (height - self.img_cropsize) // 2
                right = left + self.img_cropsize
                bottom = top + self.img_cropsize
                gt_pil = gt_pil.crop((left, top, right, bottom))
                # 转换回numpy array
                gt = np.array(gt_pil, dtype=np.uint8)
                # 重新二值化（确保是0或255）
                gt = (gt > 127).astype(np.uint8) * 255

        return img, anom_img, gt, label, img_name, img_type

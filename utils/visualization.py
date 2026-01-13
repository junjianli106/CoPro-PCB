import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

##
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

##
import matplotlib.ticker as mtick


def plot_sample_cv2(names, imgs, scores_: dict, gts, save_folder=None):
    # get subplot number
    total_number = len(imgs)

    scores = scores_.copy()
    # normarlisze anomalies
    for k, v in scores.items():
        max_value = np.max(v)
        min_value = np.min(v)

        scores[k] = (scores[k] - min_value) / max_value * 255
        scores[k] = scores[k].astype(np.uint8)
    # draw gts
    mask_imgs = []
    for idx in range(total_number):
        gts_ = gts[idx]
        mask_imgs_ = imgs[idx].copy()
        mask_imgs_[gts_ > 0.5] = (0, 0, 255)
        mask_imgs.append(mask_imgs_)

    # save imgs
    for idx in range(total_number):
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_ori.jpg'), imgs[idx])
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_gt.jpg'), mask_imgs[idx])

        for key in scores:
            heat_map = cv2.applyColorMap(scores[key][idx], cv2.COLORMAP_JET)
            visz_map = cv2.addWeighted(heat_map, 0.5, imgs[idx], 0.5, 0)
            cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_{key}.jpg'),
                        visz_map)


def plot_seg_visualization(names, imgs, scores_: dict, gts, save_folder=None, num_samples=10, seed=None, epoch=None):
    """
    可视化分割结果：随机选择指定数量的图像，显示原图、GT mask、预测分数图等
    
    Args:
        names: 图像名称列表
        imgs: 原始图像列表 (numpy array, BGR格式)
        scores_: 预测分数字典，格式为 {method_name: [score_map1, score_map2, ...]}
        gts: 真实标签mask列表
        save_folder: 保存文件夹路径
        num_samples: 随机选择的图像数量
        seed: 随机种子，用于可重复性
        epoch: epoch编号，用于在保存路径中标识
    """
    import random
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    total_number = len(imgs)
    num_samples = min(num_samples, total_number)
    
    # 随机选择图像索引
    selected_indices = random.sample(range(total_number), num_samples)
    selected_indices.sort()  # 排序以便于查看
    
    print(f'Visualizing {num_samples} randomly selected images (indices: {selected_indices})')
    
    # 确保保存文件夹存在，并在路径中包含epoch信息
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        if epoch is not None:
            vis_folder = os.path.join(save_folder, 'seg_visualization', f'ep{epoch}')
        else:
            vis_folder = os.path.join(save_folder, 'seg_visualization')
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = None
    
    scores = scores_.copy()
    # 归一化分数图
    for k, v in scores.items():
        v_array = np.array(v)
        max_value = np.max(v_array)
        min_value = np.min(v_array)
        if max_value > min_value:
            scores[k] = ((v_array - min_value) / (max_value - min_value) * 255).astype(np.uint8)
        else:
            scores[k] = v_array.astype(np.uint8)
    
    # 为每个选中的图像创建可视化
    for idx in selected_indices:
        img = imgs[idx]
        gt = gts[idx]
        name = names[idx]
        
        # 确保GT mask是2D的
        if len(gt.shape) > 2:
            gt = gt.squeeze()
        if len(gt.shape) == 2:
            gt = gt.astype(np.float32)
        else:
            # 如果GT是3D，取第一个通道
            gt = gt[:, :, 0].astype(np.float32)
        
        # 转换BGR到RGB用于matplotlib显示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建GT mask可视化（红色叠加）
        gt_mask_vis = img_rgb.copy()
        gt_binary = (gt > 0.5).astype(np.uint8)
        # 确保gt_binary是2D的
        if len(gt_binary.shape) == 2:
            gt_mask_vis[gt_binary > 0] = [255, 0, 0]  # 红色标记异常区域
        
        # 计算预测的异常区域（使用阈值）
        num_methods = len(scores)
        # 列数 = 原图(1) + GT(1) + 方法数(num_methods) = num_methods + 2
        num_cols = num_methods + 2
        fig_height = 3 if num_methods == 1 else 2.5
        fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, fig_height * 2))
        if num_cols == 1:
            axes = axes.reshape(2, -1)
        
        # 第一行：原图、GT mask、预测分数图
        # 原图
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original Image', fontsize=10)
        axes[0, 0].axis('off')
        
        # GT mask
        axes[0, 1].imshow(gt_mask_vis)
        axes[0, 1].set_title('Ground Truth Mask', fontsize=10)
        axes[0, 1].axis('off')
        
        # 预测分数图（每个方法）
        for method_idx, (method_name, score_maps) in enumerate(scores.items(), start=2):
            score_map = score_maps[idx]
            
            # 热力图
            heat_map = cv2.applyColorMap(score_map, cv2.COLORMAP_JET)
            heat_map_rgb = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
            axes[0, method_idx].imshow(heat_map_rgb)
            axes[0, method_idx].set_title(f'{method_name} Heatmap', fontsize=10)
            axes[0, method_idx].axis('off')
        
        # 第二行：叠加图
        # 原图（第二行第一列）
        axes[1, 0].imshow(img_rgb)
        axes[1, 0].set_title('Original Image', fontsize=10)
        axes[1, 0].axis('off')
        
        # GT叠加图
        gt_overlay = img_rgb.copy().astype(np.float32)
        # 创建红色mask用于叠加
        red_mask = np.zeros_like(img_rgb, dtype=np.float32)
        red_mask[:, :, 0] = gt_binary.astype(np.float32) * 255  # 红色通道
        # 叠加
        gt_overlay = cv2.addWeighted(gt_overlay, 0.7, red_mask, 0.3, 0)
        axes[1, 1].imshow(gt_overlay.astype(np.uint8))
        axes[1, 1].set_title('GT Overlay', fontsize=10)
        axes[1, 1].axis('off')
        
        # 预测叠加图（每个方法）
        for method_idx, (method_name, score_maps) in enumerate(scores.items(), start=2):
            score_map = score_maps[idx]
            
            # 归一化到0-1
            score_norm = score_map.astype(np.float32) / 255.0
            
            # 创建叠加图
            heat_map = cv2.applyColorMap(score_map, cv2.COLORMAP_JET)
            heat_map_rgb = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_rgb, 0.5, heat_map_rgb, 0.5, 0)
            axes[1, method_idx].imshow(overlay)
            axes[1, method_idx].set_title(f'{method_name} Overlay', fontsize=10)
            axes[1, method_idx].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        if vis_folder is not None:
            save_path = os.path.join(vis_folder, f'{name}_vis.jpg')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()
    
    print(f'Visualization saved to: {vis_folder}')


def plot_anomaly_score_distributions(scores: dict, ground_truths_list, save_folder, class_name):
    ground_truths = np.stack(ground_truths_list, axis=0)

    N_COUNT = 100000

    for k, v in scores.items():
        layer_score = np.stack(v, axis=0)
        normal_score = layer_score[ground_truths == 0]
        abnormal_score = layer_score[ground_truths != 0]

        plt.clf()
        plt.figure(figsize=(4, 3))
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # with plt.style.context(['science', 'ieee', 'no-latex']):
        sns.histplot(np.random.choice(normal_score, N_COUNT), color="green", bins=50, label='${d(p_n)}$',
                     stat='probability', alpha=.75)
        sns.histplot(np.random.choice(abnormal_score, N_COUNT), color="red", bins=50, label='${d(p_a)}$',
                     stat='probability', alpha=.75)

        plt.xlim([0, 3])

        save_path = os.path.join(save_folder, f'distributions_{class_name}_{k}.jpg')

        plt.savefig(save_path, bbox_inches='tight', dpi=300)


valid_feature_visualization_methods = ['TSNE', 'PCA']


def visualize_feature(features, labels, legends, n_components=3, method='TSNE'):
    assert method in valid_feature_visualization_methods
    assert n_components in [2, 3]

    if method == 'TSNE':
        model = TSNE(n_components=n_components)
    elif method == 'PCA':
        model = PCA(n_components=n_components)

    else:
        raise NotImplementedError

    feat_proj = model.fit_transform(features)

    if n_components == 2:
        ax = scatter_2d(feat_proj, labels)
    elif n_components == 3:
        ax = scatter_3d(feat_proj, labels)
    else:
        raise NotImplementedError

    plt.legend(legends)
    plt.axis('off')


def scatter_3d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes(projection='3d')

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter3D(feat_proj[label == l, 0],
                      feat_proj[label == l, 1],
                      feat_proj[label == l, 2], s=5)

    return ax1


def scatter_2d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes()

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter(feat_proj[label == l, 0],
                    feat_proj[label == l, 1], s=5)

    return ax1

import numpy as np
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def calculate_max_f1(gt, scores):
    precision, recall, thresholds = precision_recall_curve(gt, scores)
    a = 2 * precision * recall
    b = precision + recall
    f1s = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    index = np.argmax(f1s)
    max_f1 = f1s[index]
    threshold = thresholds[index]
    return max_f1, threshold


def metric_cal_img(img_scores, gt_list, map_scores=None):
    # calculate image-level ROC AUC score
    max_map_scores = map_scores.reshape(map_scores.shape[0], -1).max(axis=1)

    img_scores = 1.0 / (1.0 / max_map_scores + 1.0 / img_scores)
    # img_scores = img_scores

    gt_list = np.asarray(gt_list, dtype=int)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)

    result_dict = {'i_roc': img_roc_auc * 100}

    return result_dict


#
def metric_cal_pix(map_scores, gt_mask_list, max_samples=1000000):
    """
    计算像素级 ROC AUC

    Args:
        map_scores: 分数图，shape (N, H, W)
        gt_mask_list: 真实标签掩码列表
        max_samples: 最大采样像素数，超过此数量将进行采样以加速计算。如果为None，则不采样，使用全部数据
    """
    gt_mask = np.asarray(gt_mask_list, dtype=int)
    map_scores = np.asarray(map_scores, dtype=np.float32)

    # 检查GT mask的值范围（仅用于验证，不输出）
    unique_gt_values = np.unique(gt_mask)

    # 统计正负样本数量（仅用于验证，不输出）
    positive_pixels = np.sum(gt_mask > 0)
    total_pixels = gt_mask.size

    # 如果正样本比例异常（>50%或<0.1%），发出警告
    positive_ratio = positive_pixels / total_pixels
    if positive_ratio > 0.5:
        print(f'WARNING: Positive pixel ratio is very high ({positive_ratio:.2%}), GT mask may be incorrect!')
    elif positive_ratio < 0.001:
        print(f'WARNING: Positive pixel ratio is very low ({positive_ratio:.2%}), GT mask may be incorrect!')

    # 展平数据
    gt_flat = gt_mask.flatten()
    scores_flat = map_scores.flatten()

    total_pixels = len(gt_flat)

    # 如果max_samples为None，不采样，使用全部数据
    if max_samples is None:
        # 使用全部数据计算ROC AUC
        print('Computing ROC AUC on full data (no sampling)...')
        per_pixel_rocauc = roc_auc_score(gt_flat, scores_flat)
        print(f'ROC AUC computed successfully.')
    # 如果像素点数量超过阈值，进行采样以加速计算
    elif total_pixels > max_samples:
        print(f'Pixel count ({total_pixels:,}) exceeds max_samples ({max_samples:,}), using stratified sampling...')

        # 优化：使用更高效的分层采样方法，避免创建完整的索引数组
        # 先统计正负样本数量（快速操作）
        positive_mask = gt_flat > 0
        n_positive = positive_mask.sum()
        n_negative = total_pixels - n_positive

        # 计算采样比例
        if n_positive > 0:
            # 保持正负样本比例，但限制总样本数
            positive_ratio = n_positive / total_pixels

            # 计算采样数量，确保不超过 max_samples
            if n_positive <= max_samples:
                # 正样本数量不多，全部保留
                n_sample_positive = n_positive
                n_sample_negative = min(n_negative, max_samples - n_sample_positive)
            else:
                # 正样本数量也很多，需要采样
                # 保持正负样本比例进行采样
                n_sample_positive = min(n_positive, int(max_samples * positive_ratio))
                n_sample_negative = min(n_negative, max_samples - n_sample_positive)

            # 确保采样数量为正数
            n_sample_positive = max(1, n_sample_positive)  # 至少保留1个正样本
            n_sample_negative = max(0, n_sample_negative)  # 负样本可以为0

            # 高效采样策略：根据正样本比例选择最优方法
            positive_ratio_actual = n_positive / total_pixels

            # 如果正样本比例很小（<0.01），使用随机索引采样（避免创建大数组）
            # 否则使用where+shuffle方法（虽然创建数组但更可靠）
            if positive_ratio_actual < 0.01 and n_sample_positive < n_positive:
                # 使用随机索引采样正样本（适合稀疏正样本）
                sampled_positive_indices = set()
                # 估算需要的尝试次数：如果正样本比例是p，期望尝试次数约为 n_sample / p
                expected_attempts = int(n_sample_positive / positive_ratio_actual * 1.5)  # 1.5倍安全系数
                max_attempts = min(expected_attempts, total_pixels)  # 不超过总像素数

                while len(sampled_positive_indices) < n_sample_positive and len(
                        sampled_positive_indices) < max_attempts:
                    idx = np.random.randint(0, total_pixels)
                    if positive_mask[idx]:
                        sampled_positive_indices.add(idx)

                sampled_positive_indices = np.array(list(sampled_positive_indices), dtype=np.int64)

                # 如果随机采样不够，回退到完整索引方法
                if len(sampled_positive_indices) < n_sample_positive:
                    positive_indices = np.where(positive_mask)[0]
                    np.random.shuffle(positive_indices)
                    sampled_positive_indices = positive_indices[:n_sample_positive]
            elif n_sample_positive < n_positive:
                # 使用where+shuffle方法（适合正样本较多的情况）
                positive_indices = np.where(positive_mask)[0]
                np.random.shuffle(positive_indices)
                sampled_positive_indices = positive_indices[:n_sample_positive]
            else:
                # 全部正样本：必须创建完整索引
                sampled_positive_indices = np.where(positive_mask)[0]

            # 高效采样负样本（负样本通常很多，使用随机索引更高效）
            if n_sample_negative > 0:
                if n_sample_negative < n_negative:
                    # 使用随机索引采样负样本
                    sampled_negative_indices = set()
                    negative_ratio = n_negative / total_pixels
                    expected_attempts = int(n_sample_negative / negative_ratio * 1.2)  # 负样本多，1.2倍足够
                    max_attempts = min(expected_attempts, total_pixels)

                    while len(sampled_negative_indices) < n_sample_negative and len(
                            sampled_negative_indices) < max_attempts:
                        idx = np.random.randint(0, total_pixels)
                        if not positive_mask[idx]:
                            sampled_negative_indices.add(idx)

                    sampled_negative_indices = np.array(list(sampled_negative_indices), dtype=np.int64)

                    # 如果随机采样不够，回退到完整索引方法
                    if len(sampled_negative_indices) < n_sample_negative:
                        negative_indices = np.where(~positive_mask)[0]
                        np.random.shuffle(negative_indices)
                        sampled_negative_indices = negative_indices[:n_sample_negative]
                else:
                    # 全部负样本：必须创建完整索引
                    sampled_negative_indices = np.where(~positive_mask)[0]

                # 合并采样索引
                sampled_indices = np.concatenate([sampled_positive_indices, sampled_negative_indices])
            else:
                # 只有正样本
                sampled_indices = sampled_positive_indices

            # 使用采样后的数据
            gt_sampled = gt_flat[sampled_indices]
            scores_sampled = scores_flat[sampled_indices]

            print(
                f'Sampled pixels: {len(sampled_indices):,} (positive: {len(sampled_positive_indices):,}, negative: {n_sample_negative:,})')
        else:
            # 如果没有正样本，只采样负样本
            n_sample_negative = min(n_negative, max_samples)
            if n_sample_negative > 0:
                # 负样本很多，使用随机索引采样
                if n_sample_negative < n_negative:
                    sampled_indices = set()
                    negative_ratio = n_negative / total_pixels
                    expected_attempts = int(n_sample_negative / negative_ratio * 1.2)
                    max_attempts = min(expected_attempts, total_pixels)

                    while len(sampled_indices) < n_sample_negative and len(sampled_indices) < max_attempts:
                        idx = np.random.randint(0, total_pixels)
                        if gt_flat[idx] == 0:
                            sampled_indices.add(idx)

                    sampled_indices = np.array(list(sampled_indices), dtype=np.int64)

                    # 如果随机采样不够，回退到完整索引方法
                    if len(sampled_indices) < n_sample_negative:
                        negative_indices = np.where(gt_flat == 0)[0]
                        np.random.shuffle(negative_indices)
                        sampled_indices = negative_indices[:n_sample_negative]
                else:
                    # 全部负样本
                    sampled_indices = np.where(gt_flat == 0)[0]

                gt_sampled = gt_flat[sampled_indices]
                scores_sampled = scores_flat[sampled_indices]
                print(f'Sampled pixels: {len(sampled_indices):,} (all negative)')
            else:
                # 这种情况不应该发生，但为了安全起见
                raise ValueError("No samples to compute ROC AUC")

        print('Computing ROC AUC on sampled data...')
        # 计算 ROC AUC
        per_pixel_rocauc = roc_auc_score(gt_sampled, scores_sampled)
        print(f'ROC AUC computed successfully.')
    else:
        # 像素点数量不多，直接计算
        per_pixel_rocauc = roc_auc_score(gt_flat, scores_flat)

    # pro_auc_score = cal_pro_metric(gt_mask_list, map_scores, fpr_thresh=0.3)

    result_dict = {'p_roc': per_pixel_rocauc * 100}

    return result_dict


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def cal_pro_metric(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score


def calculate_max_f1_region(labeled_imgs, score_imgs, pro_thresh=0.6, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    # labeled_imgs[labeled_imgs <= 0.1] = 0
    # labeled_imgs[labeled_imgs > 0.1] = 1
    labeled_imgs = labeled_imgs.astype(bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    f1_list = []
    recall_list = []
    precision_list = []

    binary_score_maps = np.zeros_like(score_imgs, dtype=bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap

        predict_region_number = 0
        gt_region_number = 0

        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)

            score_map = measure.label(binary_score_maps[i], connectivity=2)
            score_props = measure.regionprops(score_map)

            predict_region_number += len(score_props)
            gt_region_number += len(props)

            # if len(score_props) == 0 or len(props) == 0:
            #     pro.append(0)
            #     continue

            for score_prop in score_props:
                x_min_0, y_min_0, x_max_0, y_max_0 = score_prop.bbox
                cur_pros = [0]
                for prop in props:
                    x_min_1, y_min_1, x_max_1, y_max_1 = prop.bbox

                    x_min = min(x_min_0, x_min_1)
                    y_min = min(y_min_0, y_min_1)
                    x_max = max(x_max_0, x_max_1)
                    y_max = max(y_max_0, y_max_1)

                    cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                    cropped_gt_label = labeled_imgs[i][x_min:x_max, y_min:y_max]

                    # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                    # cropped_mask = prop.filled_image  # corrected!
                    intersection = np.logical_and(cropped_pred_label, cropped_gt_label).astype(np.float32).sum()
                    union = np.logical_or(cropped_pred_label, cropped_gt_label).astype(np.float32).sum()
                    cur_pros.append(intersection / union)

                pro.append(max(cur_pros))

        pro = np.array(pro)

        if gt_region_number == 0 or predict_region_number == 0:
            print(f'gt_number: {gt_region_number}, pred_number: {predict_region_number}')
            recall = 0
            precision = 0
            f1 = 0
        else:
            recall = np.array(pro >= pro_thresh).astype(np.float32).sum() / gt_region_number
            precision = np.array(pro >= pro_thresh).astype(np.float32).sum() / predict_region_number

            if recall == 0 or precision == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision)

        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)

    # as array
    f1_list = np.array(f1_list)
    max_f1 = f1_list.max()
    cor_recall = recall_list[f1_list.argmax()]
    cor_precision = precision_list[f1_list.argmax()]
    print(f'cor recall: {cor_recall}, cor precision: {cor_precision}')
    return max_f1

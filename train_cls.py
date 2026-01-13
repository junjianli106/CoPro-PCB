
import argparse
import time
import sys
import os

import torch
import torch.optim.lr_scheduler
from torch.cuda.amp import autocast, GradScaler

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *
from torchvision import transforms
import random
from tqdm import tqdm

TASK = 'CLS'

# 导入 nn 和 F 以便后续使用
import torch.nn as nn
from torch.nn import functional as F

# 导入当前版本的 CoPro
from CoPro import CoPro, TripletLoss


def save_check_point(model, path):
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

    torch.save(selected_state_dict, path)

def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        check_path: str,
        train_data: DataLoader,
        ):

    # change the model into eval mode
    model.eval_mode()

    features1 = []
    features2 = []
    features1_ab = []
    features2_ab = []
    for (data, data_anom, mask, label, name, img_type) in train_data:
        # 数据已经在DataLoader中预处理好了，直接移动到device
        # 使用non_blocking=True加速数据传输（需要配合pin_memory=True）
        data = data.to(device, non_blocking=True)
        anom_data = data_anom.to(device, non_blocking=True)

        _, _, feature_map1, feature_map2 = model.encode_image(data)
        _, _, anom_feature_map1, anom_feature_map2 = model.encode_image(anom_data)

        features1.append(feature_map1)
        features2.append(feature_map2)

        features1_ab.append(anom_feature_map1)
        features2_ab.append(anom_feature_map2)

    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)

    features1_ab = torch.cat(features1_ab, dim=0)
    features2_ab = torch.cat(features2_ab, dim=0)

    model.build_image_feature_gallery(features1, features2)
    model.build_anom_image_feature_gallery(features1_ab, features2_ab)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    # 初始化混合精度训练的scaler
    # 如果模型本身使用fp16，则禁用混合精度训练（避免冲突）
    use_amp = args.use_amp and device.startswith('cuda') and model.precision != 'fp16'
    scaler = GradScaler() if use_amp else None
    if args.use_amp and model.precision == 'fp16':
        print("Warning: Model precision is fp16, disabling mixed precision training to avoid conflicts.")

    best_result_dict = None
    i_roc = 0.0
    best_roc = 0.0
    cur_epoch = 0
    pbar = tqdm(range(args.Epoch), desc=f'Epoch: {cur_epoch}, i_roc: {i_roc:.2f}, best roc: {best_roc:.2f}')
    for epoch in pbar:
        # 记录训练开始时间
        train_start_time = time.time()
        for (iter_i,train_utils) in enumerate(train_data):
            data, data_anom, mask, label, name, img_type=train_utils
            # 数据已经在DataLoader中预处理好了，直接移动到device
            data = data.to(device)
            data_anom = data_anom.to(device)

            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()

            optimizer.zero_grad()

            # 使用混合精度训练
            with autocast(enabled=use_amp):
                normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)

                abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
                abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
                abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

                # compute mean
                mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
                mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

                loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

                cls_feature, _, _, _ = model.encode_image(data)

                cls_anom_feature, _, _, _ = model.encode_image(data_anom)
                # 使用 memory bank
                if hasattr(model, 'enqueue'):
                    model.enqueue(cls_anom_feature)

                # compute v2t loss and triplet loss
                normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
                normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)
                normal_text_features = normal_text_features / normal_text_features.norm(dim=-1, keepdim=True)

                abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
                abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
                abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
                l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
                l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))
                if epoch>=1 or iter_i>=16:
                    if args.k_shot > 1:
                        # 使用 memory bank
                        if hasattr(model, 'dequeue'):
                            samepic_features, differentpic_features = model.dequeue(abnormal_text_features_ahchor,normal_text_features_ahchor)
                            # 在autocast内部，不需要手动转换为half，autocast会自动处理
                            if not use_amp and model.precision == 'fp16':
                                samepic_features, differentpic_features = samepic_features.half(), differentpic_features.half()
                            l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
                            l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

                            l_anom_pos = torch.einsum('nc,cm->nm', samepic_features, abnormal_text_features_ahchor.transpose(0, 1))
                            l_anom_pos_d = torch.einsum('nc,cm->nm', differentpic_features, abnormal_text_features_ahchor.transpose(0, 1))
                            l_anno_neg_v2t = torch.einsum('nc,cm->nm', samepic_features, normal_text_features.transpose(0, 1))
                            l_anno_neg_v2t_d = torch.einsum('nc,cm->nm', differentpic_features, normal_text_features.transpose(0, 1))

                            if model.precision == 'fp16' and not use_amp:
                                logit_scale = model.model.logit_scale.half()
                            else:
                                logit_scale = model.model.logit_scale if hasattr(model.model, 'logit_scale') else model.model.logit_scalef

                            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
                            anom_logits_v2t = torch.cat([l_anom_pos, l_anno_neg_v2t], dim=-1) * logit_scale
                            anom_logits_v2t_d = torch.cat([l_anom_pos_d, l_anno_neg_v2t_d], dim=-1) * logit_scale


                            target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)
                            target_anom_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

                            loss_v2t = criterion(logits_v2t, target_v2t)
                            loss_anom_v2t = criterion(anom_logits_v2t, target_anom_v2t)
                            loss_anom_v2t_d = criterion(anom_logits_v2t_d, target_anom_v2t)
                            loss_v2t = loss_v2t + (loss_anom_v2t + loss_anom_v2t_d)* args.lambda1
                            # loss_v2t = loss_v2t + loss_anom_v2t * args.lambda1

                            trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
                            # trip_anom_loss = criterion_tip(cls_anom_feature, abnormal_text_features_ahchor, normal_text_features_ahchor)
                            # trip_loss = trip_loss + trip_anom_loss * args.lambda1
                            trip_anom_loss = criterion_tip(samepic_features, abnormal_text_features_ahchor, normal_text_features_ahchor)
                            trip_anom_loss_d=criterion_tip(differentpic_features, abnormal_text_features_ahchor, normal_text_features_ahchor)
                            trip_loss = trip_loss + (trip_anom_loss+ trip_anom_loss_d) * args.lambda1

                            # loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1
                            loss = loss_v2t + loss_match_abnormal * args.lambda1
                    elif args.k_shot == 1:
                        # 使用 memory bank
                        if hasattr(model, 'dequeue'):
                            samepic_features = model.dequeue(abnormal_text_features_ahchor,normal_text_features_ahchor)
                            # 在autocast内部，不需要手动转换为half，autocast会自动处理
                            if not use_amp and model.precision == 'fp16':
                                samepic_features = samepic_features.half()
                                            
                            l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
                            l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

                            l_anom_pos = torch.einsum('nc,cm->nm', samepic_features, abnormal_text_features_ahchor.transpose(0, 1))
                            l_anno_neg_v2t = torch.einsum('nc,cm->nm', samepic_features, normal_text_features.transpose(0, 1))

                            if model.precision == 'fp16' and not use_amp:
                                logit_scale = model.model.logit_scale.half()
                            else:
                                logit_scale = model.model.logit_scale if hasattr(model.model, 'logit_scale') else model.model.logit_scalef

                            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
                            anom_logits_v2t = torch.cat([l_anom_pos, l_anno_neg_v2t], dim=-1) * logit_scale


                            target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)
                            target_anom_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

                            loss_v2t = criterion(logits_v2t, target_v2t)
                            loss_anom_v2t = criterion(anom_logits_v2t, target_anom_v2t)
                            loss_v2t = loss_v2t + loss_anom_v2t * args.lambda1

                            trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
                            # trip_anom_loss = criterion_tip(cls_anom_feature, abnormal_text_features_ahchor, normal_text_features_ahchor)
                            trip_anom_loss = criterion_tip(samepic_features, abnormal_text_features_ahchor, normal_text_features_ahchor)
                            trip_loss = trip_loss + trip_anom_loss * args.lambda1

                            loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1
                else:
                    l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
                    l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

                    l_anom_pos = torch.einsum('nc,cm->nm', cls_anom_feature, abnormal_text_features_ahchor.transpose(0, 1))
                    l_anno_neg_v2t = torch.einsum('nc,cm->nm', cls_anom_feature, normal_text_features.transpose(0, 1))

                    if model.precision == 'fp16' and not use_amp:
                        logit_scale = model.model.logit_scale.half()
                    else:
                        logit_scale = model.model.logit_scale if hasattr(model.model, 'logit_scale') else model.model.logit_scalef

                    logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
                    anom_logits_v2t = torch.cat([l_anom_pos, l_anno_neg_v2t], dim=-1) * logit_scale


                    target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)
                    target_anom_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

                    loss_v2t = criterion(logits_v2t, target_v2t)
                    loss_anom_v2t = criterion(anom_logits_v2t, target_anom_v2t)
                    loss_v2t = loss_v2t + loss_anom_v2t * args.lambda1

                    trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
                    
                    trip_anom_loss = criterion_tip(cls_anom_feature, abnormal_text_features_ahchor, normal_text_features_ahchor)
                    trip_loss = trip_loss + trip_anom_loss * args.lambda1

                    # loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1
                    loss = loss_v2t +  loss_match_abnormal * args.lambda1
            
            ## memorybank update
            # 使用混合精度训练进行反向传播
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # 记录训练结束时间
        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        scheduler.step()
        model.build_text_feature_gallery()

        # 预热期间跳过测试
        if epoch < args.warmup_epochs:
            test_time = 0.0
            # 输出epoch训练时间（预热期间不测试）
            print(f"\n{'='*70}")
            print(f"Epoch {epoch:3d} [WARMUP]")
            print(f"{'='*70}")
            print(f"  Training Time: {train_time:6.2f}s")
            print(f"  Testing: Skipped (warmup phase)")
            print(f"  Total Time:     {train_time:6.2f}s")
            print(f"{'='*70}\n")
            # 更新tqdm描述
            pbar.set_description(f'Epoch: {epoch}, i_roc: {i_roc:.2f}, best roc: {best_roc:.2f} (warmup)')
        else:
            # 记录测试开始时间
            test_start_time = time.time()
            scores_img = []
            score_maps = []
            test_imgs = []
            gt_list = []
            gt_mask_list = []
            names = []

            # 使用torch.no_grad()加速测试
            with torch.no_grad():
                # 收集所有需要的数据，最后批量处理以减少CPU-GPU传输
                data_tensors_list = []  # 保存处理后的tensor用于denormalization
                
                for (data, mask, label, name, img_type) in tqdm(dataloader, desc="testing"):
                    # 数据已经在DataLoader中预处理好了，直接移动到device
                    data = data.to(device, non_blocking=True)  # 使用non_blocking加速传输
                    
                    # 模型推理（保持在GPU上）
                    score_img, score_map = model(data, 'cls')
                    score_maps.extend(score_map)
                    scores_img.extend(score_img)
                    
                    # 保存tensor用于后续denormalization（延迟CPU转换）
                    data_tensors_list.append(data)
                    
                    # 批量处理ground truth数据（延迟CPU转换，减少数据传输）
                    names.extend(name)
                    # 批量处理label
                    if isinstance(label, torch.Tensor):
                        gt_list.extend(label.cpu().numpy().tolist())
                    else:
                        gt_list.extend([l.numpy() if hasattr(l, 'numpy') else l for l in label])
                    
                    # 批量处理mask（减少CPU-GPU传输）
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.cpu().numpy()
                        mask_np[mask_np > 0] = 1
                        gt_mask_list.extend([m for m in mask_np])
                    else:
                        for m in mask:
                            m_np = m.numpy() if hasattr(m, 'numpy') else m
                            m_np[m_np > 0] = 1
                            gt_mask_list.append(m_np)
                
                # 批量处理denormalization（减少CPU-GPU传输次数）
                # 按批次处理，避免一次性加载所有数据到内存
                if data_tensors_list:
                    for batch_data in data_tensors_list:
                        # 批量转换为CPU（按批次传输）
                        batch_cpu = batch_data.cpu()
                        # 批量denormalization
                        for i in range(batch_cpu.shape[0]):
                            test_imgs.append(denormalization(batch_cpu[i].numpy()))

            # 批量resize（在CPU上批量处理）
            test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
            result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))

            # 记录测试结束时间
            test_end_time = time.time()
            test_time = test_end_time - test_start_time

            i_roc = result_dict['i_roc']
            is_best = False
            
            if best_result_dict is None:
                save_check_point(model, check_path)
                best_result_dict = result_dict
                is_best = True
            elif best_result_dict['i_roc'] < result_dict['i_roc']:
                save_check_point(model, check_path)
                best_result_dict = result_dict
                is_best = True

            best_roc = best_result_dict['i_roc']
            cur_epoch = epoch
            
            # 输出epoch训练和测试时间以及测试结果（美观格式）
            best_mark = "★ BEST" if is_best else ""
            print(f"\n{'='*70}")
            print(f"Epoch {epoch:3d} [TEST RESULTS]")
            print(f"{'='*70}")
            print(f"  Image-AUROC:     {i_roc:6.2f}%")
            print(f"  Best Image-AUROC: {best_roc:6.2f}%  {best_mark}")
            print(f"{'-'*70}")
            print(f"  Training Time:   {train_time:6.2f}s")
            print(f"  Testing Time:    {test_time:6.2f}s")
            print(f"  Total Time:       {train_time + test_time:6.2f}s")
            print(f"{'='*70}\n")
            
            # 更新tqdm描述以显示最新的结果
            pbar.set_description(f'Epoch: {epoch}, i_roc: {i_roc:.2f}, best roc: {best_roc:.2f}')

    return best_result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device
    kwargs['prompt_desc'] = args.prompt_desc

    # prepare the experiment dir
    _, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # 创建transform（在DataLoader中使用）
    from datasets import create_transform
    transform = create_transform(kwargs['img_resize'], kwargs['img_cropsize'])

    # get the train dataloader（传入transform）
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, transform=transform, **kwargs)

    # get the test dataloader（传入transform）
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, transform=transform, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = CoPro(**kwargs)
    model = model.to(device)
    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_dataloader)

    i_roc = round(metrics['i_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Image-AUROC:{i_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa', 'deeppcb', 'reallad'])
    parser.add_argument('--class-name', '--class_name', type=str, default='carpet', dest='class_name')
    parser.add_argument('--prompt-desc', '--prompt_desc', type=str2bool, default=True, dest='prompt_desc')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
    parser.add_argument("--root-dir", type=str, default="./all_logs")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=100)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--bank", type=int, default=16)
    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.1)
    # mixed precision training
    parser.add_argument("--use-amp", type=str2bool, default=True, help="Use mixed precision training (fp16) for acceleration")
    # warmup epochs (skip testing during warmup)
    parser.add_argument("--warmup-epochs", type=int, default=40, help="Number of warmup epochs before starting testing")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
#
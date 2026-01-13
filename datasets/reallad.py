import glob
import os
import random

reallad_classes = ['pcb_reallad']

RealLAD_DIR = '/mnt/NVMe_1T/PCB/PCBPromptADLJJ/data/RealLAD_pytorch/1cls/'


def load_reallad(category, k_shot, seed):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.jpg")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.jpg")
                gt_paths = [os.path.join(gt_path, 'ground_truth', defect_type, os.path.basename(s)[:-4] + '.jpg') for s in
                            img_paths]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)  # 修复：应该是gt_paths，不是img_paths
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in reallad_classes

    test_img_path = os.path.join(RealLAD_DIR, category, 'test')
    train_img_path = os.path.join(RealLAD_DIR, category, 'train')
    ground_truth_path = os.path.join(RealLAD_DIR, category)  # GT文件在category/ground_truth/目录下

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, ground_truth_path)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)



    # Define the path for the seed file
    seed_file_dir = os.path.join('./datasets/seeds_reallad', category)
    os.makedirs(seed_file_dir, exist_ok=True)
    seed_file_path = os.path.join(seed_file_dir, f'selected_samples_per_run_{seed}.txt')

    if os.path.exists(seed_file_path):
        # If the seed file exists, check if the specific shot is already present
        with open(seed_file_path, 'r') as f:
            lines = f.readlines()

        shot_found = False
        for line in lines:
            if line.startswith(f'#{k_shot}:'):
                indices_str = line.split(':')[1].strip()
                training_indx = [int(idx) for idx in indices_str.split()]
                shot_found = True
                break

        if not shot_found:
            training_indx = random.sample(range(len(train_img_tot_paths)), k_shot)

            with open(seed_file_path, 'a') as f:
                f.write(f'#{k_shot}: ')
                f.write(' '.join(f'{idx:04d}' for idx in training_indx))
                f.write('\n')
    else:
        training_indx = random.sample(range(len(train_img_tot_paths)), k_shot)

        with open(seed_file_path, 'w') as f:
            f.write(f'#{k_shot}: ')
            f.write(' '.join(f'{idx:04d}' for idx in training_indx))
            f.write('\n')

    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in training_indx]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in training_indx]
    selected_train_tot_labels = [train_tot_labels[k] for k in training_indx]
    selected_train_tot_types = [train_tot_types[k] for k in training_indx]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)

import os
import shutil
import glob
import random
from PIL import Image
import numpy as np

def _mkdirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Define the paths
data_folder = '/path/to/data/RealLAD'
save_folder = '/path/to/data/RealLAD_pytorch/1cls'
train_good_folder = os.path.join(save_folder, 'pcb_reallad', 'train', 'good')
test_good_folder = os.path.join(save_folder, 'pcb_reallad', 'test', 'good')
test_bad_folder = os.path.join(save_folder, 'pcb_reallad', 'test', 'bad')
ground_truth_folder = os.path.join(save_folder, 'pcb_reallad', 'ground_truth', 'bad')

# Create necessary directories
_mkdirs_if_not_exists(train_good_folder)
_mkdirs_if_not_exists(test_good_folder)
_mkdirs_if_not_exists(test_bad_folder)
_mkdirs_if_not_exists(ground_truth_folder)

# Get all image paths
abnormal_types = ['HS', 'QS', 'YW', 'ZW']
normal_images = glob.glob(os.path.join(data_folder, 'OK', '**', '*.jpg'), recursive=True)
abnormal_images = glob.glob(os.path.join(data_folder, 'NG', '**', '*.jpg'), recursive=True)
ground_truth_images = glob.glob(os.path.join(data_folder, 'NG', '**', '*.png'), recursive=True)

# Split normal images into training and testing sets
random.shuffle(normal_images)
split_index = int(len(normal_images) * 0.5)
train_normal_images = normal_images[:split_index]
test_normal_images = normal_images[split_index:]

# Copy normal images to train and test folders
for img_path in train_normal_images:
    shutil.copy(img_path, train_good_folder)

for img_path in test_normal_images:
    shutil.copy(img_path, test_good_folder)

# Copy abnormal images and their ground truth masks to test folders
for img_path in abnormal_images:
    img_name = os.path.basename(img_path)
    mask_path = img_path.replace('.jpg', '.png')
    if mask_path in ground_truth_images:
        shutil.copy(img_path, test_bad_folder)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        mask_array[mask_array != 0] = 255
        mask = Image.fromarray(mask_array)
        mask.save(os.path.join(ground_truth_folder, img_name))

print("Data preparation completed.")

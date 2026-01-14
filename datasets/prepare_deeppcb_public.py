import os
import shutil
import random
from tqdm import tqdm

def _mkdirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_images(file_path, data_folder, save_folder, set_type):
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            try:
                image_path, _ = line.strip().split()
                temp_image_path = os.path.join(data_folder, image_path.replace('.jpg', '_temp.jpg'))
                test_image_path = os.path.join(data_folder, image_path.replace('.jpg', '_test.jpg'))

                if os.path.exists(temp_image_path):
                    label = 'good'
                    src_path = temp_image_path
                    dst_folder = os.path.join(save_folder, set_type, label)
                    _mkdirs_if_not_exists(dst_folder)
                    dst_path = os.path.join(dst_folder, os.path.basename(temp_image_path))
                    shutil.copyfile(src_path, dst_path)
                    # Ensure mutual exclusion
                    if os.path.exists(test_image_path):
                        continue
            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Exception: {e}")

def process_images_test(file_path, data_folder, save_folder, set_type):
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            try:
                image_path, _ = line.strip().split()
                temp_image_path = os.path.join(data_folder, image_path.replace('.jpg', '_temp.jpg'))
                test_image_path = os.path.join(data_folder, image_path.replace('.jpg', '_test.jpg'))

                if os.path.exists(temp_image_path) and random.random() < 0.5:
                    label = 'good'
                    src_path = temp_image_path
                    dst_folder = os.path.join(save_folder, set_type, label)
                    _mkdirs_if_not_exists(dst_folder)
                    dst_path = os.path.join(dst_folder, os.path.basename(temp_image_path))
                    shutil.copyfile(src_path, dst_path)
                    # Ensure mutual exclusion
                    if os.path.exists(test_image_path):
                        continue
                elif os.path.exists(test_image_path):
                    label = 'bad'
                    src_path = test_image_path
                    dst_folder = os.path.join(save_folder, set_type, label)
                    _mkdirs_if_not_exists(dst_folder)
                    dst_path = os.path.join(dst_folder, os.path.basename(test_image_path))
                    shutil.copyfile(src_path, dst_path)
                    # Ensure mutual exclusion
                    if os.path.exists(temp_image_path):
                        continue
            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Exception: {e}")

# Define the paths
data_folder = '/path/to/data/DeepPCB'
save_folder = '/path/to/DeepPCB_pytorch/data/1cls'
test_file = os.path.join(data_folder, 'test.txt')
trainval_file = os.path.join(data_folder, 'trainval.txt')

# Process the images
process_images_test(test_file, data_folder, save_folder, 'test')
process_images(trainval_file, data_folder, save_folder, 'train')
'''
1. DeepPCB数据集的路径如下：group20085/20085/20085000_temp.jpg （_temp.jpg 为无缺陷的模版）或者group20085/20085/20085000_test.jpg（_test.jpg和无缺陷模版对照的有缺陷的图片）。现在有很多这样的图片，我们需要将这些图片按照类别划分到不同的文件夹中。group20085和20085这一级别也有很多目录。
2. 有一个test.txt文件和trainval.txt文件（行格式为group20085/20085/20085000.jpg group20085/20085_not/20085000.txt），分别记录了有缺陷的图片的路径。我们需要将这些图片按照类别划分到不同的文件夹中。测试集还是在测试集中，训练集还是在训练集中(注意txt文件中没有_temp.jpg和_test.jpg标识，因为每个图像都有无缺陷的模版和有缺陷的图片)。
3. 一定要满足这个：将_temp.jpg放在good文件夹中，将_test.jpg放在bad文件夹中，如果这个图片的模版放到了good文件夹中，那么这个图片的_test.jpg则去掉，不需要放入到任何文件夹。如果这个图片的_test.jpg放到了bad文件夹中，那么这个图片的_temp.jpg则去掉，不需要放入到任何文件夹。互斥的关系。(请不要再原始目录删除文件，DeepPCB_pytorch不生成即可)
4. 这个数据没有ground_truth像素级别的标注，所以我们只需要将图片按照类别划分到不同的文件夹中即可。
'''
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
data_folder = '/home/lijunjian/PromptAD/data/DeepPCB'
save_folder = '/home/lijunjian/PromptAD/data/DeepPCB_pytorch/1cls'
test_file = os.path.join(data_folder, 'test.txt')
trainval_file = os.path.join(data_folder, 'trainval.txt')

# Process the images
process_images_test(test_file, data_folder, save_folder, 'test')
process_images(trainval_file, data_folder, save_folder, 'train')
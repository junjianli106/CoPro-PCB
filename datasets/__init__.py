import numpy as np
from pandas.core.config_init import use_inf_as_na_doc
from sympy.codegen.fnodes import use_rename
from torch.utils.data import DataLoader
from loguru import logger
from torchvision.models.segmentation.lraspp import LRASPPHead
from torchvision import transforms
from PIL import Image

from .dataset import CLIPDataset, CLIPDatasetWAnom
from .mvtec import load_mvtec, mvtec_classes
from .visa import load_visa, visa_classes
from .deeppcb import load_deeppcb, deeppcb_classes
from .reallad import load_reallad, reallad_classes

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

def _convert_to_rgb(image):
    return image.convert('RGB')

def create_transform(img_resize, img_cropsize):
    """创建图像预处理transform"""
    return transforms.Compose([
        transforms.Resize((img_resize, img_resize), Image.BICUBIC),
        transforms.CenterCrop(img_cropsize),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])

load_function_dict = {
    'visa': load_visa,
    'deeppcb': load_deeppcb,
    'reallad': load_reallad
}

dataset_classes = {
    'visa': visa_classes,
    'deeppcb': deeppcb_classes,
    'reallad': reallad_classes
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase, **kwargs):
    # 创建transform（如果提供了transform参数则使用，否则创建新的）
    if 'transform' in kwargs and kwargs['transform'] is not None:
        transform = kwargs['transform']
    else:
        transform = create_transform(kwargs['img_resize'], kwargs['img_cropsize'])

    if phase == 'train':
        use_anom=True
    else:
        use_anom=False
    if use_anom:
        dataset_inst = CLIPDatasetWAnom(
            load_function=load_function_dict[kwargs['dataset']],
            category=kwargs['class_name'],
            phase=phase,
            k_shot=kwargs['k_shot'],
            seed=kwargs['seed'],
            transform=transform,
            img_resize=kwargs.get('img_resize'),
            img_cropsize=kwargs.get('img_cropsize')
        )
    else:
        dataset_inst = CLIPDataset(
            load_function=load_function_dict[kwargs['dataset']],
            category=kwargs['class_name'],
            phase=phase,
            k_shot=kwargs['k_shot'],
            seed=kwargs['seed'],
            transform=transform,
            img_resize=kwargs.get('img_resize'),
            img_cropsize=kwargs.get('img_cropsize')
        )

    if phase == 'train':
        data_loader = DataLoader(
            dataset_inst, 
            batch_size=kwargs['batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True,  # 加速CPU到GPU的数据传输
            persistent_workers=True,  # 减少worker启动开销
            prefetch_factor=2  # 预取更多数据
        )
    else:
        data_loader = DataLoader(
            dataset_inst, 
            batch_size=kwargs['batch_size'], 
            shuffle=False,
            num_workers=16,
            pin_memory=True,  # 加速CPU到GPU的数据传输
            persistent_workers=True,  # 减少worker启动开销
            prefetch_factor=2  # 预取更多数据
        )

    return data_loader, dataset_inst
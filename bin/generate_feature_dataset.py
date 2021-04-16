import glob         # 用于在函数内部，对函数外面的数据进行操作
import torch
from torch.autograd import Variable
import os           # 用于处理文件和目录的函数接口
import pandas as pd # 数据库管理模块
import argparse     # 用于编写用户命令行接口的库
import numpy as np  # 数组与矩阵运算的库
import cv2          # 用于在跟踪过程中，显示跟踪结果。
import time         # 用于转换时间和日期的模块
import sys          # 与python运行环境的变量和函数有关的模块 可以获取正在运行的命令行参数和列表。
from glob import glob
from scipy import io
from siamfc.custom_transforms import ToTensor
from siamfc.alexnet import SiameseAlexNet
import torchvision.transforms as transforms

sys.path.append(os.getcwd()) #用于返回当前工作目录。


from fire import Fire   # python中用于生成命令行界面CLIs的工具，只需要在从主模块中调用fire.Fire()即可将相应代码转换为CLI。其参数可以为任何python对象
from tqdm import tqdm   # 用于显示处理进度。

#from siamfc import SiamFCTracker
# sys.path.append(r'D:\workspace\SiamTracker\SiamTrackers\SiamTrackers\SiamFC\SiamFC\siamfc\')
# from  tracker import SiamFCTracker
from siamfc import SiamFCTracker

data_dir = r'../../train_data/ILSVRC_VID_CURATION'
output_dir = r'../../train_data/Feature_Data'
model_path = r'../models/siamfc_39.pth'


def worker(output_dir, video_dir,model):
    transform = transforms.Compose([ToTensor()])

    image_names = glob(os.path.join(video_dir, '*.jpg'))
    image_names = sorted(image_names, key=lambda x: int(x.split('\\')[-1].split('.')[0]))  # 按照文件名从小到大进行排列
    video_name = video_dir.split('\\')[-1]

    save_folder = os.path.join(output_dir, video_name) # 构建输出路径文件夹
    if not os.path.exists(save_folder): # 如果输出路径文件夹不存在，则创建该文件夹。
        os.mkdir(save_folder)

    for image_name in image_names: # 遍历视频序列中的图像
        im = cv2.imread(image_name) # 裁剪视频序列中的图像
        im = im[64:191, 64:191]
        im = transform(im)[None, :, :, :]
        with torch.cuda.device(0):
            im_var = Variable(im.cuda())
            im_feature = model.features(im_var) # 获得样本的特征图
        # im_feature.numpy()  image_name.split('\\')[-1].split('.jpg')[0]
        image_name_str = image_name.split('\\')[-1].split('.jpg')[0]
        feature_img_name = os.path.join(save_folder, image_name_str + ".mat")
        io.savemat(feature_img_name,\
                   {'data': im_feature.data.cpu().squeeze().numpy()})


def feature_dataset():
    video_names = glob(data_dir + '/*')
    video_names = [x for x in video_names if os.path.isdir(x)]  # 获取Annotation Imageset和Annotation三个文件夹

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.cuda.device(0):
        model = SiameseAlexNet(0, train=False)
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
        model.eval()
    for video_dir in tqdm(video_names):
        worker(output_dir, video_dir, model)


if __name__ == '__main__':
    feature_dataset()





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
import torch.nn.functional as F
from scipy import io
from siamfc.custom_transforms import ToTensor
from siamfc.alexnet import SiameseAlexNet
import torchvision.transforms as transforms
from fire import Fire   # python中用于生成命令行界面CLIs的工具，只需要在从主模块中调用fire.Fire()即可将相应代码转换为CLI。其参数可以为任何python对象
from tqdm import tqdm   # 用于显示处理进度。
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn', True)
import functools

# 对所有视频中的特征，按照相似度求一个具有代表性的特征中心。

# 读取目录下的所有样本
data_dir = r'../../train_data/Feature_Data'
output_dir = r'../../train_data/Feature_Video'


# 将一个视频中 目标在每一帧下的特征衰减为一个特征。
def video_feature(video_dir):

    feature_names = glob(os.path.join(video_dir, '*.mat'))  # 获取该视频文件下的所有视频特征文件路径
    feature_names = sorted(feature_names, key=lambda x: int(x.split('.')[-3])) # 按照文件名称从小到大排序

    video_name = feature_names[0].split('\\')[-2]

    save_folder = os.path.join(output_dir, video_name)  # 构建输出路径文件夹
    if not os.path.exists(save_folder):  # 如果输出路径文件夹不存在，则创建该文件夹。
        os.mkdir(save_folder)


    # 获取该视频特征下的所有类别数量
    targets_index = [int(x.split('.')[-3])  for x in feature_names]
    targets_index = np.mat(targets_index)

    # 查找出target_index 中的视频序列名称，并构建新的list 用于索引。
    targets_list = []
    for i in range(targets_index.size):
        x = targets_index[0,i]
        if x not in targets_list:
            targets_list.append(x)

    ind_sum = 0 # 用于读取数据时跟进目录序列。
    for target in targets_list:  # 对每一个样本分别进行处理
        target_index = targets_index == target# 找到当前目标对应的索引
        target_feature_names = feature_names[ind_sum:(ind_sum + target_index.sum())]
        ind_sum = ind_sum + target_index.sum()

        # 根据当前目标的大小，对数据进行筛选，防止运行时间过长
        if len(target_feature_names) > 40:
            target_feature_names = target_feature_names\
                [0:len(target_feature_names):int(np.floor(len(target_feature_names)/40))]
        if len(target_feature_names) < 4:
            continue

        # 构建协方差矩阵来存储所有样本之间的相似程度
        target_cov = np.zeros([len(target_feature_names),len(target_feature_names)])
        for i in range(len(target_feature_names)):
            for j in range(i, len(target_feature_names)):
                im_i = io.loadmat(target_feature_names[i])# 从数据集中加载样本特征
                im_j = io.loadmat(target_feature_names[j])
                im_i = im_i['data']
                im_j = im_j['data']
                im_i = Variable(torch.tensor(im_i).unsqueeze(0).cuda())
                im_j = Variable(torch.tensor(im_j).unsqueeze(0).cuda())
                score = F.conv2d(im_i, im_j, groups=1) # 计算样本卷积后的得分
                target_cov[i,j] = score
                target_cov[j,i] = score
        # 根据协方差矩阵度量样本的相似程度
        target_cov_sum = target_cov.sum(axis=0)
        load_index = target_cov_sum.argsort()[-int(len(target_cov_sum)*0.95):][::-1]
        simplts_num = len(load_index)
        target_feature = np.zeros([256, 6, 6])
        for i in load_index:
            im = io.loadmat(target_feature_names[i])
            im = im['data']
            im = im/simplts_num
            target_feature = target_feature + im # 累加通过筛选的样本，并加到视频特征中

        # 将样本帧相似度排名的前95%的特征的均值作为该视频的特征输出。



        # 根据阈值，筛选样本，本保留特定阈值下的样本。
        # video_feature_index = target_cov_sum > 8000*len(target_feature_names)
        # video_feature_index[0] = True  # 第一帧一定是正确样本
        # simplts_num = video_feature_index.sum()
        # target_feature = np.zeros([256, 6, 6])
        # load_index = 0
        # for i in video_feature_index:
        #     if i:   # 如果当前样本被选中
        #         im = io.loadmat(target_feature_names[load_index])
        #         im = im['data']
        #         im = im/simplts_num
        #         target_feature = target_feature + im # 累加通过筛选的样本，并加到视频特征中
        #     load_index = load_index + 1



        # 导出当前视频的特征
        feature_name_str = target_feature_names[0].split('.')[-3]
        video_feature_name = os.path.join(save_folder, feature_name_str + ".mat")
        io.savemat(video_feature_name,\
                   {'data': target_feature}
        )
    return 1


def video_feature_dataset(data_dir):
    video_names = glob(data_dir + '/*') # 获取样本集下的所有视频目录
    if not os.path.exists(output_dir):  # 创建输出目录
        os.mkdir(output_dir)


    pool = Pool(processes=2)
    with tqdm(total=len(video_names)) as t:
        for a in pool.imap_unordered(video_feature, video_names):
            t.update()

    # for video_dir in tqdm(video_names):  # 对每一个视频序列进行处理
    #     video_feature(video_dir)


if __name__ == '__main__':
    video_feature_dataset(data_dir)








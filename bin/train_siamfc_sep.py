import os
import sys

sys.path.append(os.getcwd())
import argparse
from fire import Fire
from siamfc import train_sep

# 在命令行模式运行的时候不能开启一下命令
# import multiprocessing
# multiprocessing.set_start_method('spawn',True)

gpu_id = 0

# data_dir=r'./train_data/ILSVRC_VID_CURATION'
# model_path = r'../models/siamfc_39.pth'

# model_gpu=nn.DataParallel(model,device_ids=[0,1])# 多GPU并行计算

# output=model_gpu(input)

if __name__ == '__main__':
    # Fire(train) # Command Line Interfaces  生成命令行接口
    data_dir = r'../../train_data/ILSVRC_VID_CURATION'
    model_path = r'../models/siamfc_39.pth'
    meta_data_paths = []

    for i in range(6,7):
        meta_data_paths.append('../cluster_result/cluster' + str(i) + '_meta_data')
    # 训练
    for meta_data_path in meta_data_paths:
        train_sep(gpu_id, data_dir, meta_data_path, model_path)





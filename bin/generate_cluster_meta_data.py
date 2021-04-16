import lmdb                # 快速存取映射数据库。
import cv2                 # OpenCV 运行库
import numpy as np         # python 数学运算包
import os
import hashlib             # 摘要算法库，提供各种摘要算法，用于为数据生成摘要，可以检测到数据是否被修改。
import functools           # 用于高阶函数（将已有的函数转换为其他函数）
import argparse            # 用于解析命令行的模块
import multiprocessing
from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool
from scipy import io
import pandas as pd
import pickle  # 用于将python运行的过程中所生成的对象，在磁盘上以文件的形式存储，或者将存储的对象读取出来。

multiprocessing.set_start_method('spawn', True) # 该函数可用于设置启动进程的方式。需要注意的是，该函数的调用位置，必须位于所有与多进程有关的代码之前，此处设置为spawn可以在window和类unix哦铭泰上执行。


def worker(video_name, video_dir):
    target = {}
    video_name_dir= video_name[0].split('\\')[-2]
    target_name = video_name[0].split('\\')[-1].split('.')[0]
    image_names = glob(os.path.join(video_dir, video_name_dir, '*.' + target_name + '.x.jpg'))
    image_names = [x.split('\\')[-1].split('.')[0] for x in image_names]
    target[int(target_name)] = image_names
    return (video_name_dir, target)

def generate_cluster_meta_data(cluster_dir, video_dir, num_threads):
    # 从聚类结果中读出 每个视频序列对应的类别。
    cluster_index = pd.read_table(os.path.join(cluster_dir, 'result_index.txt'), header=None)
    cluster_index = cluster_index.to_numpy()
    cluster_root = pd.read_table(os.path.join(cluster_dir, 'result_root.txt'), header=None)
    cluster_root = cluster_root.values.tolist()

    # 根据cluster_index 将cluster_root 中相同的类别索引出来。
    # 构建一个词典用于存储这些类别。
    cluster = {}
    cluster_num = np.max(cluster_index)
    for i in range(1, cluster_num + 1):
        dict_name = 'cluster' + str(i)
        dict_value = [root for index, root in zip(cluster_index, cluster_root) if index==i]
        cluster[dict_name] = dict_value

    # 从每一个聚类结果中，提取对应的视频序列，并将该视频序列存储为lmdb文件。
    for key in cluster.keys():
        video_names = cluster[key]  # 获取该类别下，所有视频序列的名称
        meta_data = []
        # 解析对应的视频序列，并将用于训练的视频序列存储于lmdb文件中。
        # 解析对应的视频序列

        # # 采用多线程的方式生成lmdb
        # with Pool(processes = num_threads) as pool:
        #     for ret in tqdm(pool.imap_unordered(\
        #             functools.partial(worker,video_dir = r'../../SiamFC_dataset/train_data/ILSVRC_VID_CURATION'),\
        #             video_names), total=len(video_names)):
        #         with db.begin(write=True) as txn:
        #             for k, v in ret.items():
        #                 txn.put(k, v)

        # 采用单线程的方式生成lmdb。

        for video_name in tqdm(video_names):
            video_data = worker(video_name, video_dir)
            meta_data.append(video_data)
        # 存储当前cluster的meta data.
        path = os.path.join(cluster_dir, key + '_meta_data')
        f = open(path, 'wb')
        pickle.dump(meta_data, f)
        f.close()
# 获取存储的数据 a = pickle.load(open('E:\workspace\SiamFC\cluster_result\cluster1_meta_data','rb'))



cluster_dir = r'../cluster_result'
video_dir = r'../../train_data/ILSVRC_VID_CURATION'
num_threads = 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # 创建一个ArgumentParser对象，该对象包含将命令行解析称python数据类型所需的全部信息。

    # 添加参数/添加属性
    # 在给属性名之前加上”--“就能够将之变为可选参数。
    parser.add_argument('--d', default=cluster_dir, type=str, help='聚类结果存放的位置，包含聚类结果的索引，每个样本对应的目录')
    parser.add_argument('--v', default=video_dir, type=str, help="存储视频序列的目录")
    parser.add_argument('--n', default=num_threads, type=int, help="并行提取数据集时对应的线程数量")
    # 将属性返回到实例中。 将parser中所设置的所有 add_argument返回到args子类实例中， 那么parser中增加的耐属性内容都会在args实例中。
    args = parser.parse_args()

    generate_cluster_meta_data(args.d, args.v, args.n)




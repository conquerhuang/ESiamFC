import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from .custom_transforms import ToTensor

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn
from siamfc import SiameseFeature1, SiameseFeature2

from .config import config


class SiameseAlexNetSep(nn.Module):
    def __init__(self, gpu_id, train=True, model_path=''):
        super(SiameseAlexNetSep, self).__init__()

        # 加载干网络分支
        self.feature1 = SiameseFeature1()
        self.feature2 = SiameseFeature2()
        self.corr_bias = nn.Parameter(torch.zeros(1))  # Parameter
        # 预加载网络权重
        if train:
            self._load_model(self.feature1, model_path)
            self._load_model(self.feature2, model_path)
            # 固定干分支部分的网络权重
            for p in self.feature1.parameters():
                p.requires_grad = False
            self.corr_bias = nn.Parameter(torch.load(model_path)['corr_bias'])

        if train:  # 下杠+函数名 警告是私有函数，但是外部依然可以调用  train_response_sz=15  训练阶段的响应图 15x15
            gt, weight = self._create_gt_mask((config.train_response_sz, config.train_response_sz))
            with torch.cuda.device(gpu_id):
                self.train_gt = torch.from_numpy(gt).cuda()
                self.train_weight = torch.from_numpy(weight).cuda()  # train_gt.dtype=torch.float32
            gt, weight = self._create_gt_mask((config.response_sz, config.response_sz))  # response_sz=17 验证阶段的响应图 17x17
            with torch.cuda.device(gpu_id):  # 验证集 17x17
                self.valid_gt = torch.from_numpy(gt).cuda()
                self.valid_weight = torch.from_numpy(weight).cuda()
        self.exemplar = None  # 范例
        self.gpu_id = gpu_id

    ''' self.modules() ,self.children() https://blog.csdn.net/dss_dssssd/article/details/83958518
        modules 包含了网络的本身和所有的子模块（深度优先遍历）  children只包含了第二层的网络
    '''

    def features(self, x):
        out1 = self.feature1(x)
        out = self.feature2(out1)
        return out

    def _load_model(self, model, model_path):
        saved_model = torch.load(model_path)
        model_dict = model.state_dict()
        state_dict = {}
        if 'feature1.0.weight' in model_dict.keys():  # 加载self.feature1中的参数
            for k, v in saved_model.items():
                if k.replace('features', 'feature1') in model_dict.keys():
                    state_dict[k.replace('features', 'feature1')] = v
        if 'feature2.0.weight' in model_dict.keys():  # 加载self.feature2中的参数
            for k, v in saved_model.items():
                if k.split('.')[0] == 'features':
                    if int(k.split('.')[1])>6:
                        state_dict['feature2.' + str(int((k.split('.')[1]))-8) + '.' + '.'.join(k.split('.')[2:])] = v

        print('updated keys in stem net:{}'.format(state_dict.keys()))
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        pass
        return model

    def init_weights(self):
        pass
        # for m in self.modules():  # 第一次返回的是网络本身 m=SiameseAlexNet
        #     '''
        #      #关于isinstance和type的比较 https://www.runoob.com/python/python-func-isinstance.html
        #      type不会认为子类是一种父类类型，不考虑继承关系.
        #      isinstance 会认为子类是一种父类类型，会考虑继承关系
        #     '''
        #     if isinstance(m, nn.Conv2d):  # fan_in 保持weights的方差在前向传播中不变，使用fan_out保持weights的方差在反向传播中不必变
        #         nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')  # 正态分布
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)  # pytorch中，一般来说，如果对tensor的一个函数后加上了下划线，则表明这是一个in-place类型
        #         m.bias.data.zero_()  # app

    def forward(self, x):
        exemplar, instance = x  # x = ( exemplar, instance )
        # 训 练 阶 段
        if exemplar is not None and instance is not None:  #
            batch_size = exemplar.shape[0]  #
            exemplar = self.features(exemplar)  # 8 ，256， 6 ， 6
            instance = self.features(instance)  # 8,  256， 20, 20
            score_map = []
            N, C, H, W = instance.shape
            instance = instance.view(1, -1, H, W)
            score = F.conv2d(instance, exemplar, groups=N) * config.response_scale + self.corr_bias
            return score.transpose(0, 1)  # 0和1维度交换
        # 测 试 阶 段（第一帧初始化）
        elif exemplar is not None and instance is None:  # 初始化的时候只输入了exemplar
            # inference used     提取模板特征
            self.exemplar = self.features(exemplar)  # 1，256, 6, 6
            self.exemplar = torch.cat([self.exemplar for _ in range(3)], dim=0)  # [3,256,6,6]
        # 测 试 阶 段 （非第一帧）
        else:
            # inference used we don't need to scale the response or add bias
            instance = self.features(instance)  # 提取搜索区域的特征，包含三个尺度
            N, _, H, W = instance.shape
            instance = instance.view(1, -1, H, W)  # 1，NxC，H， W
            score = F.conv2d(instance, self.exemplar, groups=N)
            return score.transpose(0, 1)

    def loss(self, pred):
        return F.binary_cross_entropy_with_logits(pred, self.gt)

    def weighted_loss(self, pred):
        if self.training:
            return F.binary_cross_entropy_with_logits(pred, self.train_gt,
                                                      self.train_weight,
                                                      reduction='sum') / config.train_batch_size  # normalize the batch_size
        else:
            return F.binary_cross_entropy_with_logits(pred, self.valid_gt,
                                                      self.valid_weight,
                                                      reduction='sum') / config.valid_batch_size  # normalize the batch_size

    # 生成样本的标签和标签的权重
    def _create_gt_mask(self, shape):
        # same for all pairs
        h, w = shape  # shape=[15,15]  255x255对应17x17   (255-2*8)x(255-2*8) 对应 15x15
        y = np.arange(h, dtype=np.float32) - (h - 1) / 2.  # [0,1,2...,14]-(15-1)/2-->y=[-7, -6 ,...0,...,6,7]
        x = np.arange(w, dtype=np.float32) - (w - 1) / 2.  # [0,1,2...,14]-(15-1)/2-->x=[-7, -6 ,...0,...,6,7]
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x ** 2 + y ** 2)  # ||u-c|| 代表到中心点的距离
        mask = np.zeros((h, w))
        mask[dist <= config.radius / config.total_stride] = 1  # ||u-c||×total_stride <= radius  距离小于radius的位置是1
        mask = mask[np.newaxis, :, :]  # np.newaxis 意思是在这一个维度再增加一个一维度 mask.shape=(1,15,15)
        weights = np.ones_like(mask)  # np.ones_like 生成一个全是1的和mask一样维度
        weights[mask == 1] = 0.5 / np.sum(mask == 1)  # 0.5/num(positive sample)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)  # 0.5/num(negative sample)
        mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :,
               :]  # np.repeat(mask,batch,axis=0)数组重复三次
        return mask.astype(np.float32), weights.astype(np.float32)  # mask.shape=(8,1,15,15) train_num_workers=0的情况

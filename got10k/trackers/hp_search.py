from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k, ExperimentOTB, ExperimentUAV123, ExperimentVOT,\
    ExperimentDTB70, ExperimentNfS, ExperimentLaSOT, ExperimentTColor128

from siamfc import SiamFCTrackerMixCw
import cv2
import numpy as np
from easydict import EasyDict as edict
import os
import json
from multiprocessing import Pool
import scipy.io as scio


class SiamFC_Tracker(Tracker):
    def __init__(self, model_path, name='SiameseFC_mix_cw', is_deterministic=False, hp_config=None):
        super().__init__(name=name, is_deterministic=is_deterministic)
        self.model_path = model_path
        self.hp_config = hp_config

    def init(self, image, box):
        self.box = box
        # self.model_path = model_path
        self.gpu_id = 0
        self.trakcer = SiamFCTrackerMixCw(self.model_path, self.gpu_id, hp_config=self.hp_config)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.trakcer.init(img, box)

    def update(self, image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.trakcer.update(img)
        # return self.box


def evaluate(model_path, tracker_name, hp_config):
    tracker = SiamFC_Tracker(name = tracker_name, is_deterministic=False, model_path=model_path, hp_config=hp_config)

    experiment_otb = ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version='otb2015')
    experiment_otb.run(tracker, visualize=False)
    experiment_otb.report([tracker.name])
    #
    # experiment_otb = ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version='tb50')
    # experiment_otb.run(tracker, visualize=False)
    # experiment_otb.report([tracker.name])
    #
    # experiment_otb = ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version='otb2013')
    # experiment_otb.run(tracker, visualize=False)
    # experiment_otb.report([tracker.name])
    #
    # experiment_vot = ExperimentVOT(root_dir=r'E:\dataset\tracker_evaluate_dataset\vot2016',version=2016)
    # experiment_vot.run(tracker, visualize=False)
    # experiment_vot.report([tracker.name])
    #
    # experiment_uav123 = ExperimentUAV123(root_dir=r'E:\dataset\tracker_evaluate_dataset\UAV123', version='UAV123')
    # experiment_uav123.run(tracker, visualize=False)
    # experiment_uav123.report([tracker.name])

    # experiment_dtb70 = ExperimentDTB70(root_dir=r'E:\dataset\tracker_evaluate_dataset\DTB70')
    # # experiment_dtb70.run(tracker, visualize=False)
    # experiment_dtb70.report([tracker.name])
    #
    # experiment_tcolor128 = ExperimentTColor128(root_dir=r'E:\dataset\tracker_evaluate_dataset\Tcolor128')
    # experiment_tcolor128.run(tracker, visualize=False)
    # experiment_tcolor128.report([tracker.name])
    #
    # experiment_nfs = ExperimentNfS(root_dir=r'E:\dataset\tracker_evaluate_dataset\NFS')
    # experiment_nfs.run(tracker, visualize=False)
    # experiment_nfs.report([tracker.name])

    # experiment_lasot = ExperimentLaSOT(root_dir=r'E:\dataset\tracker_evaluate_dataset\LaSOT')
    # experiment_lasot.run(tracker, visualize=False)
    # experiment_lasot.report([tracker.name])


if __name__ == '__main__':

    # 跟踪器一些超参数设置
    model_path = r'../../models/siamfc_mix_cw7.pth'
    tracker_name = 'siamfc_mix_cw_hp_search'

    win_inf =  np.array([0.1, 0.4])
    ins_size = np.array([255])
    scal_ste = np.array([1.01, 1.2])
    scal_lr =  np.array([0.1,  0.7])
    scal_pen = np.array([0.90, 0.99])

    # window_influences = [x for x in np.linspace(win_inf.min(), win_inf.max(), 5)]  # (余弦)窗函数影响程度
    # instance_sizes = 255                                                           # 搜索区域x的尺寸大小
    # scale_steps = [x for x in np.linspace(scal_ste.min(), scal_ste.max(), 5)]      # 尺度搜索的步进
    # scale_lr = [x for x in np.linspace(scal_lr.min(), scal_lr.max(), 5)]           # 尺度更新速率
    # scale_penalty = [x for x in np.linspace(scal_pen.min(), scal_pen.max(), 5)]    # 尺度更新惩罚

    hp_config = edict()
    hp_config.window_influence = 0.176
    hp_config.instance_size = 255
    hp_config.scale_step = 1.0375
    hp_config.scale_lr = 0.59
    hp_config.scale_penalty = 0.9745

    accum_result = []  # 从开始搜索到现在得到的所有结果。
    best_score_sep = []
    best_score = 0
    scores = np.array([])  # 存储每个遍历参数下的得分。
    # 调试window_influence参数
    scale_steps = [x for x in np.linspace(1.010, 1.0650, 21)]
    for scale_step in scale_steps:
        hp_config.scale_step = scale_step
        evaluate(model_path, tracker_name, hp_config)

        with open(os.path.join('./reports/OTBotb2015', tracker_name, 'performance.json')) as fp:
            result = json.load(fp)
            successScore = result[list(result.keys())[0]]['overall']['success_score']
            precisionScore = result[list(result.keys())[0]]['overall']['precision_score']
            successRate = result[list(result.keys())[0]]['overall']['success_rate']
            accum_result.append([
                hp_config.window_influence,
                hp_config.instance_size,
                hp_config.scale_step,
                hp_config.scale_lr,
                hp_config.scale_penalty,
                successScore,
                precisionScore,
                successRate
            ])
            scio.savemat('accum_result.mat', {'accum_result':accum_result})
        #  按照对这三个参数的重视程度，设置权重，并根据权重挑选最好的结果
        score = successRate*0.5 + precisionScore*0.4 + successScore*0.1
        if score > best_score:
            best_score = score
            best_score_sep = [successRate, precisionScore, successScore]
        scores = np.append(scores, np.array(score))
    # 当前参数搜索完成，更新当前参数列表
    print('best window influence:{}'.format(scale_steps[scores.argmax()]))
    print('successRate:{:.3f}   precisionScore:{:.3f}   successScore:{:.3f}'.format(*best_score_sep))



    # accum_result = []  # 从开始搜索到现在得到的所有结果。
    #
    # for i in range(1, 6):   # 进行三次迭代搜索
    #     print('iteration:{}'.format(i))
    #     print('evaluate window influence')
    #     scores = np.array([])  # 存储每个遍历参数下的得分。
    #     best_score_sep = []
    #     best_score = 0
    #     for window_influence in window_influences:   # 遍历窗影响程度超参数，并选择最佳的遍历参数，更新当前超参数。
    #         hp_config.window_influence = window_influence
    #         evaluate(model_path, tracker_name, hp_config)
    #         with open(os.path.join('./reports/OTBtb50', tracker_name, 'performance.json')) as fp:
    #             result = json.load(fp)
    #             successScore = result[list(result.keys())[0]]['overall']['success_score']
    #             precisionScore = result[list(result.keys())[0]]['overall']['precision_score']
    #             successRate = result[list(result.keys())[0]]['overall']['success_rate']
    #             accum_result.append([
    #                 hp_config.window_influence,
    #                 hp_config.instance_size,
    #                 hp_config.scale_step,
    #                 hp_config.scale_lr,
    #                 hp_config.scale_penalty,
    #                 successScore,
    #                 precisionScore,
    #                 successRate
    #             ])
    #             scio.savemat('accum_result.mat', {'accum_result':accum_result})
    #         #  按照对这三个参数的重视程度，设置权重，并根据权重挑选最好的结果
    #         score = successRate*0.5 + precisionScore*0.4 + successScore*0.1
    #         if score > best_score:
    #             best_score = score
    #             best_score_sep = [successRate, precisionScore, successScore]
    #         scores = np.append(scores, np.array(score))
    #     # 当前参数搜索完成，更新当前参数列表
    #     print('best window influence:{}'.format(window_influences[scores.argmax()]))
    #     print('successRate:{:.3f}   precisionScore:{:.3f}   successScore:{:.3f}'.format(*best_score_sep))
    #     hp_config.window_influence = window_influences[scores.argmax()]
    #     score_max1 = window_influences[scores.argmax()]
    #     scores[scores.argmax()] = 0
    #     score_max2 = window_influences[scores.argmax()]
    #     window_influences = [x for x in np.linspace(score_max1, score_max2, 5)]
    #     print('next search window influence range:{:.3f}   {:.3f}   {:.3f}   {:.3f}   {:.3f}   '.format(
    #         window_influences[0], window_influences[1], window_influences[2],
    #         window_influences[3], window_influences[4]
    #     ))


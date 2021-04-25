from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k, ExperimentOTB, ExperimentUAV123, ExperimentVOT,\
    ExperimentDTB70, ExperimentNfS, ExperimentLaSOT, ExperimentTColor128

from siamfc import SiamFCTrackerMixCw
import cv2
import numpy as np
from multiprocessing import Pool


class SiamFC_Tracker(Tracker):
    def __init__(self, model_path, name='SiameseFC_mix_cw', is_deterministic=False):
        super().__init__(name=name, is_deterministic=is_deterministic)
        self.model_path = model_path

    def init(self, image, box):
        self.box = box
        # self.model_path = model_path
        self.gpu_id = 0
        self.trakcer = SiamFCTrackerMixCw(self.model_path, self.gpu_id)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.trakcer.init(img, box)

    def update(self, image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.trakcer.update(img)
        # return self.box


def evaluate(model_path, tracker_name):
    tracker = SiamFC_Tracker(name = tracker_name, is_deterministic=False, model_path=model_path)

    # experiment_otb = ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version='otb2015')
    # experiment_otb.run(tracker, visualize=False)
    # experiment_otb.report([tracker.name])
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

    experiment_lasot = ExperimentLaSOT(root_dir=r'E:\dataset\tracker_evaluate_dataset\LaSOT')
    experiment_lasot.run(tracker, visualize=False)
    experiment_lasot.report([tracker.name])




if __name__ == '__main__':

    tracker_infos = []


    model_path = r'../../models/siamfc_mix_cw7.pth'
    model_name = 'siamfc_mix_cw7'
    tracker_infos.append([model_path, model_name])

    model_path = r'../../models/siamfc_mix_cw4.pth'
    model_name = 'siamfc_mix_cw4'
    tracker_infos.append([model_path, model_name])

    model_path = r'../../models/siamfc_mix_cw27.pth'
    model_name = 'siamfc_mix_cw27'
    tracker_infos.append([model_path, model_name])

    # for i in range(43,51):
    #     model_path = '../../models/siamfc_mix_cw' + str(i) + '.pth'
    #     model_name = 'siamfc_mix_cw_' + str(i)
    #     tracker_infos.append([model_path, model_name])

    p = Pool(processes=3)
    for tracker_info in tracker_infos:
        p.apply_async(evaluate, args=tuple(tracker_info))
    p.close()
    p.join()

    # for tracker_info in tracker_infos:
    #     evaluate(tracker_info[0], tracker_info[1])


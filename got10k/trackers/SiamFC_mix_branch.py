from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k, ExperimentOTB, ExperimentUAV123, ExperimentVOT
from siamfc import SiamFCTrackerMixBranch
import cv2
import numpy as np
from multiprocessing import Pool


class SiamFC_Tracker(Tracker):
    def __init__(self, model_path, name='SiameseFC_mix', is_deterministic=False, mu = 1):
        super().__init__(name=name, is_deterministic=is_deterministic)
        self.model_path = model_path
        self.mu = mu

    def init(self, image, box):
        self.box = box
        # self.model_path = model_path
        self.gpu_id = 0
        self.trakcer = SiamFCTrackerMixBranch(self.model_path, self.gpu_id,is_deterministic=False, sigma=3, mu=self.mu, gamma=1.4)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.trakcer.init(img, box)

    def update(self, image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.trakcer.update(img)
        # return self.box


def evaluate(model_path, tracker_name, mu):
    tracker = SiamFC_Tracker(name = tracker_name, is_deterministic=False, model_path=model_path, mu=mu)

    # E:\datasets\visual_tracking_evaluate\OTB

    experiment_otb = ExperimentOTB(root_dir=r'F:\SiamFC\validation_dataset\OTB\OTB100', version='otb2015')
    # experiment_otb = ExperimentOTB(root_dir=r'E:\datasets\visual_tracking_evaluate\OTB', version='otb2015')
    experiment_otb.run(tracker, visualize=False)
    experiment_otb.report([tracker.name])


if __name__ == '__main__':

    tracker_infos = []

    model_path = '../../models/siamfc_mix.pth'
    for mu in np.arange(1.5, 2.5, 0.1):
        model_name = 'siamfc_mix_psr_' + str(mu)
        tracker_infos.append([model_path, model_name, mu])

    # for tracker_info in tracker_infos:
    #     evaluate(tracker_info[0], tracker_info[1], tracker_info[2])

    p = Pool(processes=2)
    for tracker_info in tracker_infos:
        p.apply_async(evaluate, args=tuple(tracker_info))
    p.close()
    p.join()




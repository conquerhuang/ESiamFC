from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k, ExperimentOTB, ExperimentUAV123, ExperimentVOT
from siamfc import SiamFCTracker
import cv2
import numpy as np
from multiprocessing import Pool


class SiamFC_Tracker(Tracker):
    def __init__(self, model_path, name='SiameseFC', is_deterministic=False):
        super().__init__(name=name, is_deterministic=is_deterministic)
        self.model_path = model_path

    def init(self, image, box):
        self.box = box
        # self.model_path = model_path
        self.gpu_id = 0
        self.trakcer = SiamFCTracker(self.model_path, self.gpu_id)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.trakcer.init(img, box)

    def update(self, image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.trakcer.update(img)
        # return self.box


def evaluate(model_path, tracker_name):
    tracker = SiamFC_Tracker(name = tracker_name, is_deterministic=False, model_path=model_path)

    experiment_otb = ExperimentOTB(root_dir=r'F:\SiamFC\validation_dataset\OTB\OTB100', version='otb2015')
    experiment_otb.run(tracker, visualize=False)
    experiment_otb.report([tracker.name])


if __name__ == '__main__':

    tracker_infos = []
    for i in range(20, 50):
        # model_path = '../../models/sw_siamfc_' + str(i) + '.pth'
        model_path = '../../models/siamfc_' + str(i) + '.pth'
        model_name = 'siamfc_' + str(i)
        tracker_infos.append([model_path, model_name])

    p = Pool(processes=5)
    for tracker_info in tracker_infos:
        p.apply_async(evaluate, args=tuple(tracker_info))
    p.close()
    p.join()

    # for tracker_info in tracker_infos:
    #     evaluate(tracker_info[0], tracker_info[1])


from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k, ExperimentOTB, ExperimentUAV123, ExperimentVOT,\
    ExperimentDTB70, ExperimentNfS, ExperimentLaSOT, ExperimentTColor128

from siamfc import SiamFCTrackerMixCw
import cv2
import numpy as np


tracker_names = ['SiamFC', 'siamfc_mix_cw4', 'siamfc_mix_cw7', 'siamfc_mix_cw27']
experiment_lasot = ExperimentLaSOT(root_dir=r'E:\dataset\tracker_evaluate_dataset\LaSOT')
experiment_lasot.report(tracker_names)









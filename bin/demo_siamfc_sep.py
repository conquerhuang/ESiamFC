import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2  # 用于在跟踪过程中，显示跟踪结果。
import time
import sys

sys.path.append(os.getcwd())  # 用于返回当前工作目录。

from fire import Fire
from tqdm import tqdm

# from siamfc import SiamFCTracker
# sys.path.append(r'D:\workspace\SiamTracker\SiamTrackers\SiamTrackers\SiamFC\SiamFC\siamfc\')
from siamfc import SiamFCTrackerSep


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Demo SiamFC")
    parser.add_argument('-v', '--video', default='./../video/Basketball', type=str, help="the path of video directory")
    parser.add_argument('-g', '--gpuid', default=0, type=int, help="the id of GPU")
    parser.add_argument('-m', '--model', default='../models/siamfc_cluster6.pth', type=str, help="the path of model")
    args = parser.parse_args()

    video_dir = args.video
    gpu_id = args.gpuid
    model_path = args.model

    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "img/*.jpg")),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    path = os.path.join(video_dir, r"groundtruth_rect.txt")
    gt_bboxes = pd.read_csv(path, sep='\t|,| ',
                            header=None, names=['xmin', 'ymin', 'width', 'height'],
                            engine='python')

    title = video_dir.split('/')[-1]
    # starting tracking
    tracker = SiamFCTrackerSep(model_path, gpu_id)
    for idx, frame in enumerate(frames):
        t0 = time.time()
        if idx == 0:
            bbox = gt_bboxes.iloc[0].values
            tracker.init(frame, bbox)
            # bbox = (bbox[0]-1, bbox[1]-1,
            #         bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
        else:
            bbox = tracker.update(frame)
        # bbox xmin ymin xmax ymax
        bbox = (bbox[0] - 1, bbox[1] - 1,
                bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1)
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0),
                              2)
        gt_bbox = gt_bboxes.iloc[idx].values
        gt_bbox = (gt_bbox[0], gt_bbox[1],
                   gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3])
        frame = cv2.rectangle(frame,
                              (int(gt_bbox[0] - 1), int(gt_bbox[1] - 1)),  # 0-index
                              (int(gt_bbox[2] - 1), int(gt_bbox[3] - 1)),
                              (255, 0, 0),
                              1)

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.imshow(title, frame)
        t1 = time.time()
        print('FPS:%.4f' % (1 / (t1 - t0 + 0.00001)))
        cv2.waitKey(20)


if __name__ == "__main__":
    Fire(main)

"""
CosyPose must be installed to run this.
Cached versions are made available to avoid this dependency.
"""

import argparse
from pathlib import Path
import json
from collections import defaultdict

import torch
import numpy as np
import cv2

from ...data.config import config

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

cfg = config[args.dataset]


def recursive_dict(*_):
    return defaultdict(recursive_dict)


# only save detection results from the target objects
targets_raw = json.load(open(f'data/bop/{args.dataset}/test_targets_bop19.json'))
inst_count = recursive_dict()
target_count = 0
for t in targets_raw:
    inst_count[t['scene_id']][t['im_id']][t['obj_id']] = t['inst_count']
    target_count += t['inst_count']
print('target count', target_count)

results = torch.load(f'../cosypose/local_data/results/bop-pbr--223026/dataset={args.dataset}/results.pth.tar')
preds = results['predictions']
detections = preds['maskrcnn_detections/detections']
infos = preds['maskrcnn_detections/coarse/iteration=1'].infos
times = infos.time.to_numpy()  # times for detection results only are not available
scene_ids = infos.scene_id.to_numpy()
view_ids = infos.view_id.to_numpy()
scores = infos.score.to_numpy()
obj_ids = np.array([int(label[-6:]) for label in infos.label])
bboxes = detections.bboxes.numpy()
print('det count', len(scores))

debug = args.debug
mask_all = np.zeros(len(scores), dtype=bool)
for scene_id in sorted(inst_count.keys()):
    inst_count_scene = inst_count[scene_id]
    scene_mask = scene_ids == scene_id
    for view_id in sorted(inst_count_scene.keys()):
        inst_count_view = inst_count_scene[view_id]
        view_mask = view_ids == view_id
        if debug:
            img = cv2.imread(f'bop/{args.dataset}/{cfg.test_folder}/{scene_id:06d}/{cfg.img_folder}/'
                             f'{view_id:06d}.{cfg.img_ext}')
        break_view = False
        for obj_id in sorted(inst_count_view.keys()):
            obj_mask = obj_ids == obj_id
            mask = scene_mask & view_mask & obj_mask
            arg_mask = np.argwhere(mask).reshape(-1)
            mask_all[arg_mask] = True
            if debug:
                print(f'obj_id: {obj_id}, n_targets: {inst_count_view[obj_id]}, n_est: {mask.sum()}')
                print('scores: ', scores[arg_mask])
                img_ = img.copy()
                for j, i in enumerate(arg_mask):
                    l, t, r, b = bboxes[i]
                    c = (0, 255, 0) if j < inst_count_view[obj_id] else (0, 0, 255)
                    cv2.rectangle(img_, (l, t), (r, b), c)
                    cv2.putText(img_, f'{scores[i]:.4f}', (int(l) + 2, int(b) - 2), cv2.FONT_HERSHEY_PLAIN, 1, c)
                cv2.imshow('', img_)
                key = cv2.waitKey()
                if key == ord('q'):
                    quit()
                elif key == ord('s'):
                    break_view = True
                    break
        if break_view:
            break
print('det masked count', mask_all.sum())

folder = Path('data/detection_results')
folder.mkdir(exist_ok=True)
folder = folder / args.dataset
folder.mkdir(exist_ok=True)

np.save(f'{folder}/scene_ids.npy', scene_ids[mask_all])
np.save(f'{folder}/view_ids.npy', view_ids[mask_all])
np.save(f'{folder}/scores.npy', scores[mask_all])
np.save(f'{folder}/obj_ids.npy', obj_ids[mask_all])
np.save(f'{folder}/bboxes.npy', bboxes[mask_all])
np.save(f'{folder}/times.npy', times[mask_all])

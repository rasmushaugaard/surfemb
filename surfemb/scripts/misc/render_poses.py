import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from ...data.renderer import ObjCoordRenderer
from ...data.obj import load_objs
from ...data.config import config

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('poses')
parser.add_argument('--scene_id', type=int, default=None)
parser.add_argument('--view_id', type=int, default=None)
parser.add_argument('--no-refine', action='store_true')
parser.add_argument('--no-bbox', action='store_true')
parser.add_argument('--render-all', action='store_true')
parser.add_argument('--alpha', type=float, default=0.7)

args = parser.parse_args()

dataset = args.dataset
cfg = config[dataset]
poses_fp = Path(args.poses)
name = '-'.join(poses_fp.name.split('-')[:-1])
pose_scores_fp = poses_fp.parent / f'{name.replace("-depth", "")}-poses-scores.npy'
poses = np.load(str(poses_fp))[0 if args.no_refine else 1]
pose_scores = np.load(str(pose_scores_fp))
show_bbox = not args.no_bbox

z = poses[:, 2, 3]

root = Path('data/bop') / dataset
objs, obj_ids = load_objs(root / cfg.model_folder)

datafolder = root / cfg.test_folder


def recursive_dict(*_):
    return defaultdict(recursive_dict)


inst_count = recursive_dict()
for t in json.load(open(f'data/bop/{args.dataset}/test_targets_bop19.json')):
    inst_count[t['scene_id']][t['im_id']][t['obj_id']] = t['inst_count']

detection_folder = Path('data/detection_results') / dataset
bboxes, det_obj_ids, scene_ids, det_scores, view_ids = [
    np.load(str(detection_folder / f'{name}.npy')) for name in
    ('bboxes', 'obj_ids', 'scene_ids', 'scores', 'view_ids')
]

start_idx = list(inst_count.keys()).index(args.scene_id) if args.scene_id is not None else 0
for scene_id, scene in list(inst_count.items())[start_idx:]:
    scene_folder = datafolder / f'{scene_id:06d}'
    scene_camera = json.load((scene_folder / 'scene_camera.json').open())

    print(f'scene {scene_id} has {len(scene)} views')
    start_idx = list(scene.keys()).index(args.view_id) if args.view_id is not None and scene_id == args.scene_id else 0
    for view_id, view in list(scene.items())[start_idx:]:
        print(scene_id, view_id)
        K = np.array(scene_camera[str(view_id)]['cam_K']).reshape(3, 3)
        img = cv2.imread(str(scene_folder / cfg.img_folder / f'{view_id:06d}.{cfg.img_ext}'))
        assert img is not None
        img_ = img.copy()
        h, w = img.shape[:2]

        renderer = ObjCoordRenderer(objs, w=w, h=h)
        renderer.ctx.clear()
        for obj_id, count in view.items():
            obj_idx = obj_ids.index(obj_id)
            mask = (scene_ids == scene_id) & (view_ids == view_id) & (det_obj_ids == obj_id)
            if mask.sum() > count:
                score_threshold = sorted(det_scores[mask])[-count]
                pose_score_threshold = sorted(pose_scores[mask])[-count]
            else:
                score_threshold = -np.inf
                pose_score_threshold = -np.inf
            for bbox, det_score, pose, pose_score in zip(bboxes[mask], det_scores[mask],
                                                         poses[mask], pose_scores[mask]):
                c = (0, 0, 255) if det_score < score_threshold else (0, 255, 0)
                if show_bbox:
                    l, t, r, b = bbox.round().astype(int)
                    cv2.rectangle(img, (l, t), (r, b), c)
                    cv2.putText(img, f'{obj_id}/{det_score:.3f}', (l + 2, b - 2), cv2.FONT_HERSHEY_PLAIN, 1, c)
                if pose_score >= pose_score_threshold or args.render_all:
                    R, t = pose[:3, :3], pose[:3, 3:]
                    if np.allclose(t[:, 0], (0, 0, 0)):
                        print('no pose found')
                    else:
                        renderer.render(obj_idx=obj_idx, K=K, R=R, t=t, clear=False, read=False)

        render = renderer.read().copy()
        mask = render[..., 3] != 0
        render = render[..., :3] * 0.5 + 0.5
        render_vis = np.tile(cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)[..., None], (1, 1, 3)) / 255
        # print(render_vis.shape)
        render_vis[mask] = render[mask] * args.alpha + render_vis[mask] * (1 - args.alpha)
        if show_bbox:
            for obj_id, count in view.items():
                mask = (scene_ids == scene_id) & (view_ids == view_id) & (det_obj_ids == obj_id)
                if mask.sum() > count:
                    score_threshold = sorted(pose_scores[mask])[-count]
                else:
                    score_threshold = -np.inf
                for bbox, pose_score in zip(bboxes[mask], pose_scores[mask]):
                    c = (0, 0, 1.) if pose_score < score_threshold else (0, 1., 0)
                    l, t, r, b = bbox.round().astype(int)
                    cv2.rectangle(render_vis, (l, t), (r, b), c)
                    cv2.putText(render_vis, f'{obj_id}/{pose_score:.3f}', (l + 2, b - 2), cv2.FONT_HERSHEY_PLAIN, 1, c)

        cv2.imshow('render', render_vis)
        # cv2.imshow('', img)
        next_scene = False
        while True:
            key = cv2.waitKey()
            if key == ord('q'):
                quit()
            elif key == ord('n'):
                next_scene = True
            elif key in {225, 233}:  # shift / alt
                continue
            else:
                pass  # print(key)
            break
        if next_scene:
            break

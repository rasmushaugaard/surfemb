import json
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
from tqdm import tqdm
import torch.utils.data

from .config import DatasetConfig


# BopInstanceDataset should only be used with test=True for debugging reasons
# use detector_crops.DetectorCropDataset for actual test inference


class BopInstanceDataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_root: Path, pbr: bool, test: bool, cfg: DatasetConfig,
            obj_ids: Sequence[int],
            scene_ids=None, min_visib_fract=0.1, min_px_count_visib=1024,
            auxs: Sequence['BopInstanceAux'] = tuple(), show_progressbar=True,
    ):
        self.pbr, self.test, self.cfg = pbr, test, cfg
        if pbr:
            assert not test
            self.data_folder = dataset_root / 'train_pbr'
            self.img_folder = 'rgb'
            self.depth_folder = 'depth'
            self.img_ext = 'jpg'
            self.depth_ext = 'png'
        else:
            self.data_folder = dataset_root / (cfg.test_folder if test else cfg.train_folder)
            self.img_folder = cfg.img_folder
            self.depth_folder = cfg.depth_folder
            self.img_ext = cfg.img_ext
            self.depth_ext = cfg.depth_ext

        self.auxs = auxs
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.instances = []
        if scene_ids is None:
            scene_ids = sorted([int(p.name) for p in self.data_folder.glob('*')])
        for scene_id in tqdm(scene_ids, 'loading crop info') if show_progressbar else scene_ids:
            scene_folder = self.data_folder / f'{scene_id:06d}'
            scene_gt = json.load((scene_folder / 'scene_gt.json').open())
            scene_gt_info = json.load((scene_folder / 'scene_gt_info.json').open())
            scene_camera = json.load((scene_folder / 'scene_camera.json').open())

            for img_id, poses in scene_gt.items():
                img_info = scene_gt_info[img_id]
                K = np.array(scene_camera[img_id]['cam_K']).reshape((3, 3)).copy()
                if pbr:
                    warnings.warn('Altering camera matrix, since PBR camera matrix doesnt seem to be correct')
                    K[:2, 2] -= 0.5

                for pose_idx, pose in enumerate(poses):
                    obj_id = pose['obj_id']
                    if obj_ids is not None and obj_id not in obj_ids:
                        continue
                    pose_info = img_info[pose_idx]
                    if pose_info['visib_fract'] < min_visib_fract:
                        continue
                    if pose_info['px_count_visib'] < min_px_count_visib:
                        continue

                    bbox_visib = pose_info['bbox_visib']
                    bbox_obj = pose_info['bbox_obj']

                    cam_R_obj = np.array(pose['cam_R_m2c']).reshape(3, 3)
                    cam_t_obj = np.array(pose['cam_t_m2c']).reshape(3, 1)

                    self.instances.append(dict(
                        scene_id=scene_id, img_id=int(img_id), K=K, obj_id=obj_id, pose_idx=pose_idx,
                        bbox_visib=bbox_visib, bbox_obj=bbox_obj, cam_R_obj=cam_R_obj, cam_t_obj=cam_t_obj,
                        obj_idx=obj_idxs[obj_id],
                    ))

        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        for aux in self.auxs:
            instance = aux(instance, self)
        return instance


class BopInstanceAux:
    def init(self, dataset: BopInstanceDataset):
        pass

    def __call__(self, data: dict, dataset: BopInstanceDataset) -> dict:
        pass


def _main():
    from .config import tless
    for pbr, test in (True, False), (False, False), (False, True):
        print(f'pbr: {pbr}, test: {test}')
        data = BopInstanceDataset(dataset_root=Path('bop/tless'), pbr=pbr, test=test, cfg=tless, obj_ids=range(1, 31))
        print(len(data))


if __name__ == '__main__':
    _main()

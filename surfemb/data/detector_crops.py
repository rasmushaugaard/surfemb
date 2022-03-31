import json
from pathlib import Path
from typing import Sequence
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch.utils.data

from .config import DatasetConfig
from .instance import BopInstanceAux


class DetectorCropDataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_root: Path, obj_ids, detection_folder: Path, cfg: DatasetConfig,
            auxs: Sequence[BopInstanceAux], show_progressbar=True,
    ):
        self.data_folder = dataset_root / cfg.test_folder
        self.img_folder = cfg.img_folder
        self.depth_folder = cfg.depth_folder
        self.img_ext = cfg.img_ext
        self.depth_ext = cfg.depth_ext

        self.bboxes = np.load(str(detection_folder / 'bboxes.npy'))
        self.obj_ids = np.load(str(detection_folder / 'obj_ids.npy'))
        self.scene_ids = np.load(str(detection_folder / 'scene_ids.npy'))
        self.view_ids = np.load(str(detection_folder / 'view_ids.npy'))
        self.obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}

        self.auxs = auxs
        self.instances = []
        scene_ids = sorted([int(scene_dir.name) for scene_dir in self.data_folder.glob('*') if scene_dir.is_dir()])

        self.scene_cameras = defaultdict(lambda *_: [])

        for scene_id in tqdm(scene_ids, 'loading crop info') if show_progressbar else scene_ids:
            scene_folder = self.data_folder / f'{scene_id:06d}'
            self.scene_cameras[scene_id] = json.load((scene_folder / 'scene_camera.json').open())

        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, i):
        scene_id, view_id, obj_id = self.scene_ids[i], self.view_ids[i], self.obj_ids[i]
        instance = dict(
            scene_id=scene_id, img_id=view_id, obj_id=obj_id, obj_idx=self.obj_idxs[obj_id],
            K=np.array(self.scene_cameras[scene_id][str(view_id)]['cam_K']).reshape((3, 3)),
            mask_visib=self.bboxes[i], bbox=self.bboxes[i].round().astype(int),
        )
        for aux in self.auxs:
            instance = aux(instance, self)
        return instance


def _main():
    import argparse
    import cv2
    from .config import config
    from . import std_auxs
    from .obj import load_objs

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()

    dataset_root = Path(f'bop/{args.dataset}')
    cfg = config[args.dataset]
    objs, obj_ids = load_objs(dataset_root / cfg.model_folder, None)

    data = DetectorCropDataset(
        dataset_root=dataset_root, cfg=cfg, obj_ids=obj_ids,
        detection_folder=Path(f'detection_results/{args.dataset}'),
        auxs=(
            std_auxs.RgbLoader(),
            std_auxs.RandomRotatedMaskCrop(224, max_angle=0, offset_scale=0, use_bbox=True),
        ),
    )
    while True:
        i = np.random.randint(len(data))
        img = data[i]['rgb_crop']
        cv2.imshow('', img[..., ::-1])
        if cv2.waitKey() == ord('q'):
            quit()


if __name__ == '__main__':
    _main()

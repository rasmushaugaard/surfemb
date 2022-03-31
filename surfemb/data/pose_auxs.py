from typing import Sequence

import numpy as np

from .instance import BopInstanceAux
from .obj import Obj
from .renderer import ObjCoordRenderer


class ObjCoordAux(BopInstanceAux):
    def __init__(self, objs: Sequence[Obj], res: int, mask_key='mask_visib_crop', replace_mask=False, sigma=0.):
        self.objs, self.res = objs, res
        self.mask_key = mask_key
        self.replace_mask = replace_mask
        self.renderer = None
        self.sigma = sigma

    def get_renderer(self):
        # lazy instantiation of renderer to create the context in the worker process
        if self.renderer is None:
            self.renderer = ObjCoordRenderer(self.objs, self.res)
        return self.renderer

    def __call__(self, inst: dict, _) -> dict:
        renderer = self.get_renderer()
        K = inst['K_crop'].copy()

        if self.sigma > 0:
            # offset principal axis slightly to encourage all object coordinates within the pixel to have
            # som probability mass. Smoother probs -> more robust score and better posed refinement opt. problem.
            while True:
                offset = np.random.randn(2)
                if np.linalg.norm(offset) < 3:
                    K[:2, 2] += offset * self.sigma
                    break

        obj_coord = renderer.render(inst['obj_idx'], K, inst['cam_R_obj'], inst['cam_t_obj']).copy()
        if self.mask_key is not None:
            if self.replace_mask:
                mask = obj_coord[..., 3]
            else:
                mask = obj_coord[..., 3] * inst[self.mask_key] / 255
                obj_coord[..., 3] = mask
            inst[self.mask_key] = (mask * 255).astype(np.uint8)
        inst['obj_coord'] = obj_coord
        return inst


class SurfaceSampleAux(BopInstanceAux):
    def __init__(self, objs: Sequence[Obj], n_samples: int, norm=True):
        self.objs, self.n_samples = objs, n_samples
        self.norm = norm

    def __call__(self, inst: dict, _) -> dict:
        obj = self.objs[inst['obj_idx']]
        mesh = obj.mesh_norm if self.norm else obj.mesh
        inst['surface_samples'] = mesh.sample(self.n_samples).astype(np.float32)
        return inst


class MaskSamplesAux(BopInstanceAux):
    def __init__(self, n_samples: int, mask_key='mask_visib_crop'):
        self.mask_key = mask_key
        self.n_samples = n_samples

    def __call__(self, inst: dict, _):
        mask_arg = np.argwhere(inst[self.mask_key])  # (N, 2)
        idxs = np.random.choice(np.arange(len(mask_arg)), self.n_samples, replace=self.n_samples > len(mask_arg))
        inst['mask_samples'] = mask_arg[idxs]  # (n_samples, 2)
        return inst

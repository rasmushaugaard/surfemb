import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import torch
import torch.utils.data
import trimesh
import trimesh.sample


@contextmanager
def timer(text='', do=True):
    if do:
        start = time.time()
        try:
            yield
        finally:
            print(f'{text}: {time.time() - start:.4}s')
    else:
        yield


@contextmanager
def add_timing_to_list(l):
    start = time.time()
    try:
        yield
    finally:
        l.append(time.time() - start)


def balanced_dataset_concat(a, b):
    # makes an approximately 50/50 concat
    # by adding copies of the smallest dataset
    if len(a) < len(b):
        a, b = b, a
    assert len(a) >= len(b)
    data = a
    for i in range(round(len(a) / len(b))):
        data += b
    return data


def load_surface_samples(dataset, obj_ids, root=Path('data')):
    surface_samples = [trimesh.load_mesh(root / f'surface_samples/{dataset}/obj_{i:06d}.ply').vertices for i in obj_ids]
    surface_sample_normals = [trimesh.load_mesh(root / f'surface_samples_normals/{dataset}/obj_{i:06d}.ply').vertices
                              for i in obj_ids]
    return surface_samples, surface_sample_normals


class Rodrigues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rvec):
        R, jac = cv2.Rodrigues(rvec.detach().cpu().numpy())
        jac = torch.from_numpy(jac).to(rvec.device)
        ctx.save_for_backward(jac)
        return torch.from_numpy(R).to(rvec.device)

    @staticmethod
    def backward(ctx, grad_output):
        jac, = ctx.saved_tensors
        return jac @ grad_output.to(jac.device).reshape(-1)


def rotate_batch(batch: torch.Tensor):  # (..., H, H) -> (4, ..., H, H)
    assert batch.shape[-1] == batch.shape[-2]
    return torch.stack([
        batch,  # 0 deg
        torch.flip(batch, [-2]).transpose(-1, -2),  # 90 deg
        torch.flip(batch, [-1, -2]),  # 180 deg
        torch.flip(batch, [-1]).transpose(-1, -2),  # 270 deg
    ])  # (4, ..., H, H)


def rotate_batch_back(batch: torch.Tensor):  # (4, ..., H, H) -> (4, ..., H, H)
    assert batch.shape[0] == 4
    assert batch.shape[-1] == batch.shape[-2]
    return torch.stack([
        batch[0],  # 0 deg
        torch.flip(batch[1], [-1]).transpose(-1, -2),  # -90 deg
        torch.flip(batch[2], [-1, -2]),  # -180 deg
        torch.flip(batch[3], [-2]).transpose(-1, -2),  # -270 deg
    ])  # (4, ..., H, H)


class EmptyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, item):
        return None

import argparse
from typing import Sequence, Union

import torch
from torch.nn import functional as F
import numpy as np
import wandb
import albumentations as A
import cv2
import pytorch_lightning as pl

from . import data
from .dep.unet import ResNetUNet
from .dep.siren import Siren
from .data.obj import Obj
from .data.tfms import denormalize
from . import utils

# could be extended to allow other mlp architectures
mlp_class_dict = dict(
    siren=Siren
)


class SurfaceEmbeddingModel(pl.LightningModule):
    def __init__(self, n_objs: int, emb_dim=12, n_pos=1024, n_neg=1024, lr_cnn=3e-4, lr_mlp=3e-5,
                 mlp_name='siren', mlp_hidden_features=256, mlp_hidden_layers=2,
                 key_noise=1e-3, warmup_steps=2000, separate_decoders=True,
                 **kwargs):
        """
        :param emb_dim: number of embedding dimensions
        :param n_pos: number of positive (q, k) pairs from the object mask
        :param n_neg: number of negative keys, k-, from the object surface
        """
        super().__init__()
        self.save_hyperparameters()

        self.n_objs, self.emb_dim = n_objs, emb_dim
        self.n_pos, self.n_neg = n_pos, n_neg
        self.lr_cnn, self.lr_mlp = lr_cnn, lr_mlp
        self.warmup_steps = warmup_steps
        self.key_noise = key_noise
        self.separate_decoders = separate_decoders

        # query model
        self.cnn = ResNetUNet(
            n_class=(emb_dim + 1) if separate_decoders else n_objs * (emb_dim + 1),
            n_decoders=n_objs if separate_decoders else 1,
        )
        # key models
        mlp_class = mlp_class_dict[mlp_name]
        mlp_args = dict(in_features=3, out_features=emb_dim,
                        hidden_features=mlp_hidden_features, hidden_layers=mlp_hidden_layers)
        self.mlps = torch.nn.Sequential(*[mlp_class(**mlp_args) for _ in range(n_objs)])

    @staticmethod
    def model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group(SurfaceEmbeddingModel.__name__)
        parser.add_argument('--emb-dim', type=int, default=12)
        parser.add_argument('--single-decoder', dest='separate_decoders', action='store_false')
        return parent_parser

    def get_auxs(self, objs: Sequence[Obj], crop_res: int):
        random_crop_aux = data.std_auxs.RandomRotatedMaskCrop(crop_res)
        return (
            data.std_auxs.RgbLoader(),
            data.std_auxs.MaskLoader(),
            random_crop_aux.definition_aux,
            # Some image augmentations probably make most sense in the original image, before rotation / rescaling
            # by cropping. 'definition_aux' registers 'AABB_crop' such that the "expensive" image augmentation is only
            # performed where the crop is going to be taken from.
            data.std_auxs.TransformsAux(key='rgb', crop_key='AABB_crop', tfms=A.Compose([
                A.GaussianBlur(blur_limit=(1, 3)),
                A.ISONoise(),
                A.GaussNoise(),
                data.tfms.DebayerArtefacts(),
                data.tfms.Unsharpen(),
                A.CLAHE(),  # could probably be moved to the post-crop augmentations
                A.GaussianBlur(blur_limit=(1, 3)),
            ])),
            random_crop_aux.apply_aux,
            data.pose_auxs.ObjCoordAux(objs, crop_res, replace_mask=True),
            data.pose_auxs.SurfaceSampleAux(objs, self.n_neg),
            data.pose_auxs.MaskSamplesAux(self.n_pos),
            data.std_auxs.TransformsAux(tfms=A.Compose([
                A.CoarseDropout(max_height=16, max_width=16, min_width=8, min_height=8),
                A.ColorJitter(hue=0.1),
            ])),
            data.std_auxs.NormalizeAux(),
            data.std_auxs.KeyFilterAux({'rgb_crop', 'obj_coord', 'obj_idx', 'surface_samples', 'mask_samples'})
        )

    def get_infer_auxs(self, objs: Sequence[Obj], crop_res: int, from_detections=True):
        auxs = [data.std_auxs.RgbLoader()]
        if not from_detections:
            auxs.append(data.std_auxs.MaskLoader())
        auxs.append(data.std_auxs.RandomRotatedMaskCrop(
            crop_res, max_angle=0,
            offset_scale=0 if from_detections else 1,
            use_bbox=from_detections,
            rgb_interpolation=(cv2.INTER_LINEAR,),
        ))
        if not from_detections:
            auxs += [
                data.pose_auxs.ObjCoordAux(objs, crop_res, replace_mask=True),
                data.pose_auxs.SurfaceSampleAux(objs, self.n_neg),
                data.pose_auxs.MaskSamplesAux(self.n_pos),
            ]
        return auxs

    def configure_optimizers(self):
        opt = torch.optim.Adam([
            dict(params=self.cnn.parameters(), lr=1e-4),
            dict(params=self.mlps.parameters(), lr=3e-5),
        ])
        sched = dict(
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1., i / self.warmup_steps)),
            interval='step'
        )
        return [opt], [sched]

    def step(self, batch, log_prefix):
        img = batch['rgb_crop']  # (B, 3, H, W)
        coord_img = batch['obj_coord']  # (B, H, W, 4) [-1, 1]
        obj_idx = batch['obj_idx']  # (B,)
        coords_neg = batch['surface_samples']  # (B, n_neg, 3) [-1, 1]
        mask_samples = batch['mask_samples']  # (B, n_pos, 2)

        device = img.device
        B, _, H, W = img.shape
        assert coords_neg.shape[1] == self.n_neg
        mask = coord_img[..., 3] == 1.  # (B, H, W)
        y, x = mask_samples.permute(2, 0, 1)  # 2 x (B, n_pos)

        if self.separate_decoders:
            cnn_out = self.cnn(img, obj_idx)  # (B, 1 + emb_dim, H, W)
            mask_lgts = cnn_out[:, 0]  # (B, H, W)
            queries = cnn_out[:, 1:]  # (B, emb_dim, H, W)
        else:
            cnn_out = self.cnn(img)  # (B, n_objs + n_objs * emb_dim, H, W)
            mask_lgts = cnn_out[torch.arange(B), obj_idx]  # (B, H, W)
            queries = cnn_out[:, self.n_objs:].view(B, self.n_objs, self.emb_dim, H, W)
            queries = queries[torch.arange(B), obj_idx]  # (B, emb_dim, H, W)

        mask_prob = torch.sigmoid(mask_lgts)  # (B, H, W)
        mask_loss = F.binary_cross_entropy(mask_prob, mask.type_as(mask_prob))

        queries = queries[torch.arange(B).view(B, 1), :, y, x]  # (B, n_pos, emb_dim)

        # compute similarities for positive pairs
        coords_pos = coord_img[torch.arange(B).view(B, 1), y, x, :3]  # (B, n_pos, 3) [-1, 1]
        coords_pos += torch.randn_like(coords_pos) * self.key_noise
        keys_pos = torch.stack([self.mlps[i](c) for i, c in zip(obj_idx, coords_pos)])  # (B, n_pos, emb_dim)
        sim_pos = (queries * keys_pos).sum(dim=-1, keepdim=True)  # (B, n_pos, 1)

        # compute similarities for negative pairs
        coords_neg += torch.randn_like(coords_neg) * self.key_noise
        keys_neg = torch.stack([self.mlps[i](v) for i, v in zip(obj_idx, coords_neg)])  # (B, n_neg, n_dim)
        sim_neg = queries @ keys_neg.permute(0, 2, 1)  # (B, n_pos, n_neg)

        # loss
        lgts = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)  # (B, 1 + n_neg, n_pos)
        target = torch.zeros(B, self.n_pos, device=device, dtype=torch.long)
        nce_loss = F.cross_entropy(lgts, target)

        loss = mask_loss + nce_loss
        self.log(f'{log_prefix}/loss', loss)
        self.log(f'{log_prefix}/mask_loss', mask_loss)
        self.log(f'{log_prefix}/nce_loss', nce_loss)
        return loss

    def training_step(self, batch, _):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        self.log_image_sample(batch)
        return self.step(batch, 'valid')

    def get_emb_vis(self, emb_img: torch.Tensor, mask: torch.Tensor = None, demean: torch.tensor = False):
        if demean is True:
            demean = emb_img[mask].view(-1, self.emb_dim).mean(dim=0)
        if demean is not False:
            emb_img = emb_img - demean
        shape = emb_img.shape[:-1]
        emb_img = emb_img.view(*shape, 3, -1).mean(dim=-1)
        if mask is not None:
            emb_img[~mask] = 0.
        emb_img /= torch.abs(emb_img).max() + 1e-9
        emb_img.mul_(0.5).add_(0.5)
        return emb_img

    def log_image_sample(self, batch, i=0):
        img = batch['rgb_crop'][i]
        obj_idx = batch['obj_idx'][i]
        coord_img = batch['obj_coord'][i]
        coord_mask = coord_img[..., 3] != 0

        mask_lgts, query_img = self.infer_cnn(img, obj_idx)
        query_img = self.get_emb_vis(query_img)
        mask_est = torch.tile(torch.sigmoid(mask_lgts)[..., None], (1, 1, 3))

        key_img = self.infer_mlp(coord_img[..., :3], obj_idx)
        key_img = self.get_emb_vis(key_img, mask=coord_mask, demean=True)

        log_img = torch.cat((
            denormalize(img).permute(1, 2, 0), mask_est, query_img, key_img,
        ), dim=1).cpu().numpy()
        self.trainer.logger.experiment.log(dict(
            embeddings=wandb.Image(log_img),
            global_step=self.trainer.global_step
        ))

    @torch.no_grad()
    def infer_cnn(self, img: Union[np.ndarray, torch.Tensor], obj_idx, rotation_ensemble=True):
        assert not self.training
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = data.tfms.normalize(img)
            img = torch.from_numpy(img).to(self.device)
        _, h, w = img.shape

        if rotation_ensemble:
            img = utils.rotate_batch(img)  # (4, 3, h, h)
        else:
            img = img[None]  # (1, 3, h, w)
        cnn_out = self.cnn(img, [obj_idx] * len(img) if self.separate_decoders else None)
        if not self.separate_decoders:
            channel_idxs = [obj_idx] + list(self.n_objs + obj_idx * self.emb_dim + np.arange(self.emb_dim))
            cnn_out = cnn_out[:, channel_idxs]
        # cnn_out: (B, 1+emb_dim, h, w)
        if rotation_ensemble:
            cnn_out = utils.rotate_batch_back(cnn_out).mean(dim=0)
        else:
            cnn_out = cnn_out[0]
        mask_lgts, query_img = cnn_out[0], cnn_out[1:]
        query_img = query_img.permute(1, 2, 0)  # (h, w, emb_dim)
        return mask_lgts, query_img

    @torch.no_grad()
    def infer_mlp(self, pts_norm: Union[np.ndarray, torch.Tensor], obj_idx):
        assert not self.training
        if isinstance(pts_norm, np.ndarray):
            pts_norm = torch.from_numpy(pts_norm).to(self.device).float()
        return self.mlps[obj_idx](pts_norm)  # (..., emb_dim)

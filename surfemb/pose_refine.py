import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.optimize import minimize

from .utils import Rodrigues


def refine_pose(R: np.ndarray, t: np.ndarray, query_img, renderer, obj_idx, K_crop, obj_, model, keys_verts,
                interpolation='bilinear', n_samples_denom=4096, method='BFGS'):
    """
    Refines the pose estimate (R, t) by local maximization of the log prob (according to the queries / keys)
    of the initially visible surface.
    Bilinear interpolation and PyTorch autograd to get the gradient, and BFGS for optimization.
    """
    h, w, _ = query_img.shape
    assert h == w
    res_crop = h
    device = model.device

    # Get the object coordinates and keys of the initially visible surface
    coord_img = renderer.render(obj_idx, K_crop, R, t)
    mask = coord_img[..., 3] == 1.
    coord_norm_masked = torch.from_numpy(coord_img[..., :3][mask]).to(device)  # (N, 3)
    keys_masked = model.infer_mlp(coord_norm_masked, obj_idx)  # (N, emb_dim)
    coord_masked = coord_norm_masked * obj_.scale + torch.from_numpy(obj_.offset).to(device)
    coord_masked = torch.cat((coord_masked, torch.ones(len(coord_masked), 1, device=device)), dim=1)  # (N, 4)
    K_crop = torch.from_numpy(K_crop).to(device)

    # precompute log denominator in softmax (log sum exp over keys) per query
    # needs to be batched or estimated with reduced amount of keys (as implemented here) because of memory requirements
    keys_sampled = keys_verts[torch.randperm(len(keys_verts), device=device)[:n_samples_denom]]
    denom_img = torch.logsumexp(query_img @ keys_sampled.T, dim=-1, keepdim=True)  # (H, W, 1)
    coord_masked = coord_masked.float()
    K_crop = K_crop.float()

    def sample(img, p_img_norm):
        samples = F.grid_sample(
            img.permute(2, 0, 1)[None],  # (1, d, H, W)
            p_img_norm[None, None],  # (1, 1, N, 2)
            align_corners=False,
            padding_mode='border',
            mode=interpolation,
        )  # (1, d, 1, N)
        return samples[0, :, 0].T  # (N, d)

    def objective(pose: np.ndarray, return_grad=False):
        pose = torch.from_numpy(pose).float()
        pose.requires_grad = return_grad
        Rt = torch.cat((
            Rodrigues.apply(pose[:3]),
            pose[3:, None],
        ), dim=1).to(device)  # (3, 4)

        P = K_crop @ Rt
        p_img = coord_masked @ P.T
        p_img = p_img[..., :2] / p_img[..., 2:]  # (N, 2)
        # pytorch grid_sample coordinates
        p_img_norm = (p_img + 0.5) * (2 / res_crop) - 1

        query_sampled = sample(query_img, p_img_norm)  # (N, emb_dim)
        log_nominator = (keys_masked * query_sampled).sum(dim=-1)  # (N,)
        log_denominator = sample(denom_img, p_img_norm)[:, 0]  # (N,)
        score = -(log_nominator.mean() - log_denominator.mean()) / 2

        if return_grad:
            score.backward()
            return pose.grad.detach().cpu().numpy()
        else:
            return score.item()

    rvec = cv2.Rodrigues(R)[0]
    pose = np.array((*rvec[:, 0], *t[:, 0]))
    result = minimize(fun=objective, x0=pose, jac=lambda pose: objective(pose, return_grad=True), method=method)

    pose = result.x
    R = cv2.Rodrigues(pose[:3])[0]
    t = pose[3:, None]
    return R, t, result.fun

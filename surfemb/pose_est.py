import numpy as np
import cv2
import torch
import torch_scatter
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from .utils import timer


def estimate_pose(mask_lgts: torch.tensor, query_img: torch.tensor,
                  obj_pts: torch.tensor, obj_normals: torch.tensor, obj_keys: torch.tensor, obj_diameter: float,
                  K: np.ndarray, max_poses=10000, max_pose_evaluations=1000, down_sample_scale=3, alpha=1.5,
                  dist_2d_min=0.1, pnp_method=cv2.SOLVEPNP_AP3P, pose_batch_size=500, max_pool=True,
                  avg_queries=True, do_prune=True, visualize=False, poses=None, debug=False):
    """
    Builds correspondence distribution from queries and keys,
    samples correspondences with inversion sampling,
    samples poses from correspondences with P3P,
    prunes pose hypothesis,
    and scores pose hypotheses based on estimated mask and correspondence distribution.

    :param mask_lgts: (r, r)
    :param query_img: (r, r, e)
    :param obj_pts: (m, 3)
    :param obj_normals: (m, 3)
    :param obj_keys: (m, e)
    :param alpha: exponent factor for correspondence weighing
    :param K: (3, 3) camera intrinsics
    :param max_poses: number of poses to sample (before pruning)
    :param max_pose_evaluations: maximum number of poses to evaluate / score after pruning
    :param dist_2d_min: minimum 2d distance between at least one pair of correspondences for a hypothesis
    :param max_pool: max pool probs spatially to make score more robust (but less accurate),
        similar to a reprojection error threshold in common PnP RANSAC frameworks
    :param poses: evaluate these poses instead of sampling poses
    """
    device = mask_lgts.device
    r = mask_lgts.shape[0]
    m, e = obj_keys.shape

    # down sample
    K = K.copy()
    K[:2, 2] += 0.5  # change origin to corner
    K[:2] /= down_sample_scale
    K[:2, 2] -= 0.5  # change origin back

    mask_log_prob, neg_mask_log_prob = [
        F.max_pool2d(F.logsigmoid(lgts)[None], down_sample_scale)[0]
        for lgts in (mask_lgts, -mask_lgts)
    ]
    mask_lgts = F.avg_pool2d(mask_lgts[None], down_sample_scale)[0]
    res_sampled = len(mask_lgts)
    n = res_sampled ** 2
    mask_prob = torch.sigmoid(mask_lgts).view(n)

    yy = torch.arange(res_sampled, device=device)
    yy, xx = torch.meshgrid(yy, yy)
    yy, xx = (v.reshape(n) for v in (yy, xx))
    img_pts = torch.stack((xx, yy), dim=1)  # (n, 2)

    if max_pool:
        mask_log_prob = F.max_pool2d(mask_log_prob[None], 3, 1, 1)[0]
        neg_mask_log_prob = F.max_pool2d(neg_mask_log_prob[None], 3, 1, 1)[0]
    mask_log_prob = mask_log_prob.view(n)
    neg_mask_log_prob = neg_mask_log_prob.view(n)

    with timer('corr matrix', debug):
        if avg_queries:
            queries = F.avg_pool2d(query_img.permute(2, 0, 1), down_sample_scale).view(e, n).T  # (n, e)
            corr_matrix_log = torch.log_softmax(queries @ obj_keys.T, dim=1)  # (n, m)
            corr_matrix = corr_matrix_log.exp()
        else:
            # evaluate the whole corr_matrix followed by max pool for evaluation (corr_matrix_log), batched to avoid oom
            # use kernel centers for corr_matrix
            query_img = query_img[:res_sampled * down_sample_scale, :res_sampled * down_sample_scale].permute(2, 0, 1)
            corr_matrix = torch.empty(res_sampled, res_sampled, m, device=device, dtype=torch.float32)
            corr_matrix_log = torch.empty_like(corr_matrix)
            patch_out_len = int((res_sampled * 0.5) // down_sample_scale)
            patch_len = patch_out_len * down_sample_scale
            n_patches_len = int(np.ceil(res_sampled * down_sample_scale / patch_len))
            offset = down_sample_scale // 2
            for i in range(n_patches_len):
                l, lo = patch_len * i, patch_out_len * i
                ro = lo + patch_out_len
                for j in range(n_patches_len):
                    t, to = patch_len * j, patch_out_len * j
                    bo = to + patch_out_len

                    patch = query_img[:, t:t + patch_len, l:l + patch_len]
                    shape = patch.shape[1:]
                    patch_corr_log = torch.log_softmax(patch.reshape(e, -1).T @ obj_keys.T, dim=1).view(*shape, m)
                    corr_matrix[to:bo, lo:ro] = patch_corr_log[offset::down_sample_scale, offset::down_sample_scale]
                    corr_matrix_log[to:bo, lo:ro] = F.max_pool2d(
                        patch_corr_log.permute(2, 0, 1), down_sample_scale).permute(1, 2, 0)
            corr_matrix_log = corr_matrix_log.view(n, m)
            corr_matrix = corr_matrix.view(n, m).exp_()
        corr_matrix *= mask_prob[:, None]

        if max_pool:
            # max pool spatially
            # batched over m to avoid oom
            corr_matrix_log = corr_matrix_log.view(res_sampled, res_sampled, m).permute(2, 0, 1)  # (m, rs, rs)
            m_bs = 10000
            for i in range(int(np.ceil(m / m_bs))):
                l, r = i * m_bs, (i + 1) * m_bs
                corr_matrix_log[l:r] = F.max_pool2d(corr_matrix_log[l:r], kernel_size=3, stride=1, padding=1)
            corr_matrix_log = corr_matrix_log.permute(1, 2, 0).view(n, m)

    if poses is None:
        with timer('sample corr', debug):
            corr_matrix = corr_matrix.view(-1)
            corr_matrix.pow_(alpha)
            corr_matrix_cumsum = torch.cumsum(corr_matrix, dim=0, out=corr_matrix)
            corr_matrix_cumsum /= corr_matrix_cumsum[-1].item()
            corr_matrix = None  # cumsum is overwritten. Reset variable to avoid accidental use
            corr_idx = torch.searchsorted(corr_matrix_cumsum, torch.rand(max_poses, 4, device=device))  # (max_poses, 4)
            del corr_matrix_cumsum  # frees gpu memory

            p2d_idx, p3d_idx = corr_idx.div(m, rounding_mode='floor'), corr_idx % m
            p2d, p3d = img_pts[p2d_idx].float(), obj_pts[p3d_idx]  # (max_poses, 4, 2 xy), (max_poses, 4, 3 xyz)
            n3d = obj_normals[p3d_idx[:, :3].cpu().numpy()]  # (max_poses, 3, 3 nx ny nz)

        with timer('to cpu', debug):
            p2d, p3d = p2d.cpu().numpy(), p3d.cpu().numpy()

        if visualize:
            corr_2d_vis = np.zeros((r, r))
            p2d_xx, p2d_yy = p2d.astype(int).reshape(-1, 2).T
            np.add.at(corr_2d_vis, (p2d_yy, p2d_xx), 1)
            corr_2d_vis /= corr_2d_vis.max()
            cv2.imshow('corr_2d_vis', corr_2d_vis)

        poses = np.zeros((max_poses, 3, 4))
        poses_mask = np.zeros(max_poses, dtype=bool)
        with timer('pnp', debug):
            rotvecs = np.zeros((max_poses, 3))
            for i in range(max_poses):
                ret, rvecs, tvecs = cv2.solveP3P(p3d[i], p2d[i], K, None, flags=pnp_method)
                if rvecs:
                    j = np.random.randint(len(rvecs))
                    rotvecs[i] = rvecs[j][:, 0]
                    poses[i, :3, 3:] = tvecs[j]
                    poses_mask[i] = True
            poses[:, :3, :3] = Rotation.from_rotvec(rotvecs).as_matrix()
        poses, p2d, p3d, n3d = [a[poses_mask] for a in (poses, p2d, p3d, n3d)]

        with timer('pose pruning', debug):
            # Prune hypotheses where all correspondences come from the same small area in the image
            dist_2d = np.linalg.norm(p2d[:, :3, None] - p2d[:, None, :3], axis=-1).max(axis=(1, 2))  # (max_poses,)
            dist_2d_mask = dist_2d >= dist_2d_min * res_sampled

            # Prune hypotheses that are very close to or very far from the camera compared to the crop
            z = poses[:, 2, 3]
            z_min = K[0, 0] * obj_diameter / (res_sampled * 20)
            z_max = K[0, 0] * obj_diameter / (res_sampled * 0.5)
            size_mask = (z_min < z) & (z < z_max)

            # Prune hypotheses where correspondences are not visible, estimated by the face normal.
            Rt = poses[:, :3, :3].transpose(0, 2, 1)  # (max_poses, 3, 3)
            n3d_cam = n3d @ Rt  # (max_poses, 3 pts, 3 nxnynz)
            p3d_cam = p3d[:, :3] @ Rt + poses[:, None, :3, 3]  # (max_poses, 3 pts, 3 xyz)
            normals_dot = (n3d_cam * p3d_cam).sum(axis=-1)  # (max_poses, 3 pts)
            normals_mask = np.all(normals_dot < 0, axis=-1)  # (max_poses,)

            # allow not pruning for debugging reasons
            if do_prune:
                poses = poses[dist_2d_mask & size_mask & normals_mask]  # (n_poses, 3, 4)
    else:
        dist_2d, size_mask, normals_mask = None, None, None

    poses = poses[slice(None, max_pose_evaluations)]
    n_poses = len(poses)
    R = poses[:, :3, :3]  # (n_poses, 3, 3)
    t = poses[:, :3, 3]  # (n_poses, 3)

    if debug:
        print('n_poses', n_poses)

    def batch_score(R: torch.tensor, t: torch.tensor, visualize=False):
        n_poses = len(R)
        # project to image
        obj_pts_cam = obj_pts @ R.permute(0, 2, 1) + t[:, None]  # (n_poses, m, 3)
        z = obj_pts_cam[..., 2]  # (n_poses, m)
        obj_pts_img = obj_pts_cam @ K.T
        u = (obj_pts_img[..., :2] / obj_pts_img[..., 2:]).round_()  # (n_poses, m, 2 xy)
        # ignore pts outside the image
        mask_neg = torch.any(torch.logical_or(u < 0, res_sampled <= u), dim=-1)  # (n_poses, m)
        # convert 2D-coordinates to flat indexing
        u = u[..., 1].mul_(res_sampled).add_(u[..., 0])  # (n_poses, m)
        # use an ignore bin to allow batched scatter_min
        u[mask_neg] = n  # index for the ignore bin
        # maybe u should be rounded before casting to long - or converted to long after rounding above
        # but a small test shows that there are no rounding errors
        u = u.long()

        # per pixel, find the vertex closest to the camera
        z, z_arg = torch_scatter.scatter_min(z, u, dim_size=n + 1)  # 2x(n_poses, n + 1 ignore bin)
        z, z_arg = z[:, :-1], z_arg[:, :-1]  # then discard the ignore bin: 2x(n_poses, n)
        # get mask of populated pixels
        mask = z > 0  # (n_poses, n)
        mask_pose_idx, mask_n_idx = torch.where(mask)  # 2x (k,)
        z, z_arg = z[mask_pose_idx, mask_n_idx], z_arg[mask_pose_idx, mask_n_idx]  # 2x (k,)
        u = u[mask_pose_idx, z_arg]  # (k,)

        mask_score_2d = neg_mask_log_prob[None].expand(n_poses, n).clone()  # (n_poses, n)
        mask_score_2d[mask_pose_idx, u] = mask_log_prob[u]
        mask_score = mask_score_2d.mean(dim=1)  # (n_poses,)

        coord_score = corr_matrix_log[u, z_arg]  # (k,)
        coord_score = torch_scatter.scatter_mean(coord_score, mask_pose_idx, dim_size=n_poses)  # (n_poses,)
        # handle special case, where no mask pts are in the image
        coord_score_mask = torch.ones(n_poses, dtype=torch.bool, device=device)
        coord_score_mask[mask_pose_idx] = 0
        coord_score[coord_score_mask] = -np.inf

        # normalize by max entropy
        mask_score /= np.log(2)
        coord_score /= np.log(m)

        score = mask_score + coord_score  # (n_poses,)

        if visualize:
            assert len(R) == 1
            mask_score_img = mask_score_2d[0].view(res_sampled, res_sampled).cpu().numpy()  # [mi, 0]
            mask_score_img = 1 - mask_score_img / mask_score_img.min()
            cv2.imshow('mask_score', mask_score_img)

            coord_score_img = torch.zeros(res_sampled * res_sampled, 3, device=device)
            coord_score_img[:, 2] = 1.
            coord_scores = corr_matrix_log[u, z_arg]  # (k_best)
            coord_score_img[u] = (1 - coord_scores / coord_scores.min())[:, None]
            cv2.imshow('coord_score', coord_score_img.view(res_sampled, res_sampled, 3).cpu().numpy())

        return score, mask_score, coord_score

    R, t, K = (torch.from_numpy(v).float().to(device) for v in (R, t, K))
    pose_scores = torch.empty(n_poses, device=device)
    mask_scores = torch.empty(n_poses, device=device)
    coord_scores = torch.empty(n_poses, device=device)
    scores = pose_scores, mask_scores, coord_scores
    for batch_idx in range(np.ceil(n_poses / pose_batch_size).astype(int)):
        l = pose_batch_size * batch_idx
        r = l + pose_batch_size
        with timer('batch', debug):
            batch_scores = batch_score(R[l:r], t[l:r])
        for container, items in zip(scores, batch_scores):
            container[l:r] = items

    if visualize and len(pose_scores) > 0:
        best_pose_idx = torch.argmax(pose_scores)
        batch_score(R[best_pose_idx:best_pose_idx + 1], t[best_pose_idx:best_pose_idx + 1], visualize=True)
        print('pose_score', pose_scores[best_pose_idx].item())
        print('mask_score', mask_scores[best_pose_idx].item())
        print('coord_score', coord_scores[best_pose_idx].item())

    return R, t, pose_scores, mask_scores, coord_scores, dist_2d, size_mask, normals_mask

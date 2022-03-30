import argparse
from pathlib import Path

import cv2
import torch.utils.data
import numpy as np

from .. import utils
from ..data import obj
from ..data.config import config
from ..data import instance
from ..data import detector_crops
from ..data.renderer import ObjCoordRenderer
from ..surface_embedding import SurfaceEmbeddingModel
from .. import pose_est
from .. import pose_refine

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--real', action='store_true')
parser.add_argument('--detection', action='store_true')
parser.add_argument('--i', type=int, default=0)
parser.add_argument('--device', default='cuda:0')

args = parser.parse_args()
data_i = args.i
device = torch.device(args.device)
model_path = Path(args.model_path)

model = SurfaceEmbeddingModel.load_from_checkpoint(args.model_path)
model.eval()
model.freeze()
model.to(device)

dataset = model_path.name.split('-')[0]
real = args.real
detection = args.detection
root = Path('data/bop') / dataset
cfg = config[dataset]
res_crop = 224

objs, obj_ids = obj.load_objs(root / cfg.model_folder)
renderer = ObjCoordRenderer(objs, res_crop)
assert len(obj_ids) == model.n_objs
surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)
auxs = model.get_infer_auxs(objs=objs, crop_res=res_crop, from_detections=detection)
dataset_args = dict(dataset_root=root, obj_ids=obj_ids, auxs=auxs, cfg=cfg)
if detection:
    assert args.real
    data = detector_crops.DetectorCropDataset(
        **dataset_args, detection_folder=Path(f'detection_results/{dataset}')
    )
else:
    data = instance.BopInstanceDataset(**dataset_args, pbr=not args.real, test=args.real)

# initialize opencv windows
cols = 4
window_names = 'img', 'mask_est', 'queries', 'keys', \
               'dist', 'xy', 'xz', 'yz', \
               'pose', 'mask_score', 'coord_score', 'query_norm'
for j, name in enumerate(window_names):
    row = j // cols
    col = j % cols
    cv2.imshow(name, np.zeros((res_crop, res_crop)))
    cv2.moveWindow(name, 100 + 300 * col, 100 + 300 * row)

print()
print('With an opencv window active:')
print("press 'a', 'd' and 'x'(random) to get a new input image,")
print("press 'e' to estimate pose, and 'r' to refine pose estimate,")
print("press 'g' to see the ground truth pose,")
print("press 'q' to quit.")
while True:
    print()
    print('------------ new input -------------')
    inst = data[data_i]
    obj_idx = inst['obj_idx']
    img = inst['rgb_crop']
    K_crop = inst['K_crop']
    obj_ = objs[obj_idx]
    print(f'i: {data_i}, obj_id: {obj_ids[obj_idx]}')

    with utils.timer('forward_cnn'):
        mask_lgts, query_img = model.infer_cnn(img, obj_idx)

    mask_prob = torch.sigmoid(mask_lgts)
    query_vis = model.get_emb_vis(query_img)
    query_norm_img = torch.norm(query_img, dim=-1) * mask_prob
    query_norm_img /= query_norm_img.max()
    cv2.imshow('query_norm', query_norm_img.cpu().numpy())

    dist_img = torch.zeros(res_crop, res_crop, device=model.device)

    verts_np = surface_samples[obj_idx]
    verts = torch.from_numpy(verts_np).float().to(device)
    normals = surface_sample_normals[obj_idx]
    verts_norm = (verts_np - obj_.offset) / obj_.scale
    with utils.timer('forward_mlp'):
        keys_verts = model.infer_mlp(torch.from_numpy(verts_norm).float().to(model.device), obj_idx)  # (N, emb_dim)
    keys_means = keys_verts.mean(dim=0)  # (emb_dim,)

    if not detection:
        coord_img = torch.from_numpy(inst['obj_coord']).to(device)
        key_img = model.infer_mlp(coord_img[..., :3], obj_idx)
        key_mask = coord_img[..., 3] == 1
        keys = key_img[key_mask]  # (N, emb_dim)
        key_vis = model.get_emb_vis(key_img, mask=key_mask, demean=keys_means)

    # corr vis
    uv_names = 'xy', 'xz', 'yz'
    uv_slices = slice(1, None, -1), slice(2, None, -2), slice(2, 0, -1)
    uv_uniques = []
    uv_all = ((verts_norm + 1) * (res_crop / 2 - .5)).round().astype(int)
    for uv_name, uv_slice in zip(uv_names, uv_slices):
        view_uvs_unique, view_uvs_unique_inv = np.unique(uv_all[:, uv_slice], axis=0, return_inverse=True)
        uv_uniques.append((view_uvs_unique, view_uvs_unique_inv))

    # visualize
    img_vis = img[..., ::-1].astype(np.float32) / 255
    grey = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)

    for win_name in (*uv_names, 'dist', 'pose', 'mask_score', 'coord_score', 'keys'):
        cv2.imshow(win_name, np.zeros((res_crop, res_crop)))

    cv2.imshow('img', img_vis)
    cv2.imshow('mask_est', torch.sigmoid(mask_lgts).cpu().numpy())
    cv2.imshow('queries', query_vis.cpu().numpy())
    if not detection:
        cv2.imshow('keys', key_vis.cpu().numpy())

    last_mouse_pos = 0, 0
    uv_pts_3d = []
    current_pose = None
    down_sample_scale = 3


    def mouse_cb(event, x, y, flags=0, *_):
        global last_mouse_pos
        if detection:
            return
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            return
        last_mouse_pos = x, y
        q = query_img[y, x]
        p_mask = mask_prob[y, x]

        key_probs = torch.softmax(keys @ q, dim=0)  # (N,)
        dist_img[key_mask] = key_probs / key_probs.max() * p_mask
        dist_vis = np.stack((grey, grey, dist_img.cpu().numpy()), axis=-1)
        cv2.circle(dist_vis, (x, y), 10, (1., 1., 1.), 1, cv2.LINE_AA)
        cv2.imshow('dist', dist_vis)

        vert_probs = torch.softmax(keys_verts @ q, dim=0).cpu().numpy()  # (N,)
        for uv_name, uv_slice, (view_uvs_unique, view_uvs_unique_inv) in zip(uv_names, uv_slices, uv_uniques):
            uvs_unique_probs = np.zeros(len(view_uvs_unique), dtype=vert_probs.dtype)
            np.add.at(uvs_unique_probs, view_uvs_unique_inv, vert_probs)
            prob_img = np.zeros((224, 224, 3))
            yy, xx = view_uvs_unique.T
            prob_img[yy, xx, 2] = uvs_unique_probs / uvs_unique_probs.max() * p_mask.cpu().numpy()
            prob_img[yy, xx, :2] = 0.1
            for p, c in zip(uv_pts_3d, np.eye(3)[::-1]):
                p_norm = ((p - obj_.offset) / obj_.scale)[uv_slice]
                p_uv = ((p_norm + 1) * (res_crop / 2 - .5)).round().astype(int)
                cv2.drawMarker(prob_img, tuple(p_uv[::-1]), c, cv2.MARKER_CROSS, 10)
            cv2.imshow(uv_name, prob_img[::-1])


    for name in window_names:
        cv2.setMouseCallback(name, mouse_cb)


    def debug_pose_hypothesis(R, t, obj_pts=None, img_pts=None):
        global uv_pts_3d, current_pose
        current_pose = R, t
        render = renderer.render(obj_idx, K_crop, R, t)
        render_mask = render[..., 3] == 1.
        pose_img = img_vis.copy()
        pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :3][render_mask] * 0.25 + 0.25

        if obj_pts is not None:
            colors = np.eye(3)[::-1]
            for (x, y), c in zip(img_pts.astype(int), colors):
                cv2.drawMarker(pose_img, (x, y), tuple(c), cv2.MARKER_CROSS, 10)
            uv_pts_3d = obj_pts
            mouse_cb(None, *last_mouse_pos)

        cv2.imshow('pose', pose_img)

        poses = np.eye(4)
        poses[:3, :3] = R
        poses[:3, 3:] = t
        pose_est.estimate_pose(
            mask_lgts=mask_lgts, query_img=query_img,
            obj_pts=verts, obj_normals=normals, obj_keys=keys_verts,
            obj_diameter=obj_.diameter, K=K_crop, down_sample_scale=down_sample_scale,
            visualize=True, poses=poses[None],
        )


    def estimate_pose():
        print()
        with utils.timer('pnp ransac'):
            R, t, scores, mask_scores, coord_scores, dist_2d, size_mask, normals_mask = pose_est.estimate_pose(
                mask_lgts=mask_lgts, query_img=query_img, down_sample_scale=down_sample_scale,
                obj_pts=verts, obj_normals=normals, obj_keys=keys_verts,
                obj_diameter=obj_.diameter, K=K_crop,
            )
        if not len(scores):
            print('no pose')
            return None
        else:
            R, t, scores, mask_scores, coord_scores = [a.cpu().numpy() for a in
                                                       (R, t, scores, mask_scores, coord_scores)]
            best_pose_idx = np.argmax(scores)
            R_, t_ = R[best_pose_idx], t[best_pose_idx, :, None]
            debug_pose_hypothesis(R_, t_)
            return R_, t_


    while True:
        print()
        key = cv2.waitKey()
        if key == ord('q'):
            quit()
        elif key == ord('a'):
            data_i = (data_i - 1) % len(data)
            break
        elif key == ord('d'):
            data_i = (data_i + 1) % len(data)
            break
        elif key == ord('x'):
            data_i = np.random.randint(len(data))
            break
        elif key == ord('e'):
            print('pose est:')
            estimate_pose()
        elif key == ord('g'):
            print('gt:')
            debug_pose_hypothesis(inst['cam_R_obj'], inst['cam_t_obj'])
        elif key == ord('r'):
            print('refine:')
            if current_pose is not None:
                with utils.timer('refinement'):
                    R, t, score_r = pose_refine.refine_pose(
                        R=current_pose[0], t=current_pose[1], query_img=query_img, keys_verts=keys_verts,
                        obj_idx=obj_idx, obj_=obj_, K_crop=K_crop, model=model, renderer=renderer,
                    )
                    trace = np.trace(R @ current_pose[0].T)
                    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                    print(f'refinement angle diff: {np.rad2deg(angle):.1f} deg')
                debug_pose_hypothesis(R, t)

        mouse_cb(None, *last_mouse_pos)

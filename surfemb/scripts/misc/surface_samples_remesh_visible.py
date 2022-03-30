import argparse
from pathlib import Path

import pymeshlab
import trimesh
from tqdm import tqdm

from ...data.config import config

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--face-quality-threshold', type=float, default=1e-3)
parser.add_argument('--remesh-percentage', type=float, default=5.)
args = parser.parse_args()

mesh_folder = Path('data/bop') / args.dataset / config[args.dataset].model_folder
remesh_folder = Path('data/remesh_visible') / args.dataset
remesh_folder.mkdir(exist_ok=True, parents=True)

for mesh_fp in tqdm(list(mesh_folder.glob('*.ply'))):
    remesh_fp = remesh_folder / mesh_fp.name

    print()
    print(mesh_fp)
    print()

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_fp.absolute()))
    ms.repair_non_manifold_edges_by_removing_faces()
    ms.subdivision_surfaces_midpoint(iterations=10, threshold=pymeshlab.Percentage(args.remesh_percentage))
    ms.ambient_occlusion(occmode='per-Face (deprecated)', reqviews=256)
    face_quality_array = ms.current_mesh().face_quality_array()
    minq = face_quality_array.min()
    if minq < args.face_quality_threshold:
        assert face_quality_array.max() > args.face_quality_threshold
        ms.select_by_face_quality(minq=minq, maxq=args.face_quality_threshold)
        ms.delete_selected_faces()
        ms.remove_unreferenced_vertices()
    ms.save_current_mesh(str(remesh_fp.absolute()), save_textures=False)

    area_reduction = trimesh.load_mesh(remesh_fp).area / trimesh.load_mesh(mesh_fp).area
    print()
    print(mesh_fp)
    print(f'area reduction {area_reduction}')
    print()

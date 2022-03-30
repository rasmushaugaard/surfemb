import argparse
from pathlib import Path

import pymeshlab
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--n-samples', type=int, default=int(50e3))
args = parser.parse_args()

n_samples = args.n_samples

remesh_folder = Path(f'data/remesh_visible/{args.dataset}')
assert remesh_folder.exists()
even_samples_folder = Path(f'data/surface_samples/{args.dataset}')
even_samples_folder.mkdir(exist_ok=True)
ms = pymeshlab.MeshSet()

for mesh_fp in tqdm(list(remesh_folder.glob('*.ply'))):
    ms.clear()
    samples_fp = even_samples_folder / mesh_fp.name

    print()
    print(mesh_fp)
    print()

    ms.load_new_mesh(str(mesh_fp.absolute()))
    mesh_id = ms.current_mesh_id()
    n = n_samples
    while True:
        ms.set_current_mesh(mesh_id)
        ms.poisson_disk_sampling(samplenum=n)
        n_actual_samples = ms.current_mesh().vertex_number()
        print(n_actual_samples)
        if n_actual_samples >= n_samples:
            ms.save_current_mesh(str(samples_fp.absolute()), save_vertex_normal=False, save_textures=False,
                                 save_vertex_quality=False, save_vertex_color=False, save_vertex_coord=False,
                                 save_vertex_radius=False)
            break
        else:
            ms.delete_current_mesh()
            n = n + n // 2

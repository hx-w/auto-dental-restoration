# -*- coding: utf-8 -*-

import os
from typing import List
import trimesh
import numpy as np
from scipy.io import savemat
import gradio as gr

# set package path
import sys
sys.path.append('.')
from mesh_to_sdf import mesh_to_sdf


def sample_sdfs(mesh: trimesh.Trimesh, methods: List[str], base_dir: str, sample_ratios: List[int]):
    '''
    method: ToothDIT | DIF-Net
    '''
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()

    pointcloud = mesh_to_sdf.get_surface_point_cloud(
        mesh,
        surface_point_method='sample',
        sample_point_count=500000,
        calculate_normals=True
    )

    free_pnts, sdfs = pointcloud.sample_sdf_near_surface(
        1000, True, 'normal', sphere_size=11, box=False
    )
    sdfs_data = np.concatenate([free_pnts, sdfs.reshape(-1, 1)], axis=1)
    np.random.shuffle(sdfs_data)

    sp_ratios = [rat / sum(sample_ratios) for rat in sample_ratios]

    def __clip_sample(raw_data: np.array, ratios: List[float]):
        len_t, len_r = raw_data.shape[0], len(ratios)
        max_z, min_z = np.max(raw_data[:, 2]), np.min(raw_data[:, 2])
        piece_len = (max_z - min_z) / len_r
        pieces = [
            raw_data[
                np.logical_and(
                    raw_data[:, 2] >= min_z + ind * piece_len,
                    raw_data[:, 2] < min_z + (ind + 1) * piece_len
                )
            ]
            for ind in range(len_r)
        ]
        for ind in range(len_r):
            np.random.shuffle(pieces[ind])
            pieces[ind] = pieces[ind][: min(int(len_t * ratios[ind]), pieces[ind].shape[0]), :]

        return np.concatenate(pieces, axis=0)

    for method in methods:
        if method == 'ToothDIT':
            save_path = os.path.join(base_dir, f'{method}_sdfs.npz')

            surfs_data = np.hstack([pointcloud.points, np.zeros((pointcloud.points.shape[0], 1))])
            sdfs_data = np.concatenate([sdfs_data, surfs_data], axis=0)
            surfs_data = __clip_sample(surfs_data, sp_ratios)
            sdfs_data = __clip_sample(sdfs_data, sp_ratios)

            pos = sdfs_data[sdfs_data[:, 3] >= 0]
            neg = sdfs_data[sdfs_data[:, 3] < 0]

            np.savez(save_path, pos=pos, neg=neg)

        elif method == 'DIF-Net':
            surf_save_path = os.path.join(base_dir, f'{method}_on_surface.mat')
            free_save_path = os.path.join(base_dir, f'{method}_free_space.mat')

            savemat(surf_save_path, {
                'p': __clip_sample(np.concatenate([pointcloud.points, pointcloud.normals], axis=1), sp_ratios)
            })
            savemat(free_save_path, {
                'p_sdf': __clip_sample(sdfs_data, sp_ratios)
            })
        else:
            raise gr.Error('采样错误')


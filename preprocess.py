# -*- coding: utf-8 -*-

import os
import trimesh
import numpy as np
from scipy.io import savemat
import gradio as gr

# set package path
import sys
sys.path.append('.')
from mesh_to_sdf import mesh_to_sdf


def sample_sdfs(mesh: trimesh.Trimesh, method: str, base_dir: str):
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
        500000, True, 'normal', sphere_size=11, box=False
    )
    sdfs_data = np.concatenate([free_pnts, sdfs.reshape(-1, 1)], axis=1)
    np.random.shuffle(sdfs_data)

    if method == 'ToothDIT':
        save_path = os.path.join(base_dir, f'{method}_sdfs.npz')

        surfs_data = np.hstack([pointcloud.points, np.zeros((pointcloud.points.shape[0], 1))])
        sdfs_data = np.concatenate([sdfs_data, surfs_data], axis=0)

        pos = sdfs_data[sdfs_data[:, 3] >= 0]
        neg = sdfs_data[sdfs_data[:, 3] < 0]

        np.savez(
            save_path,
            pos=pos, neg=neg,
            surf_pnts=pointcloud.points, surf_norms=pointcloud.normals
        )

    elif method == 'DIF-Net':
        surf_save_path = os.path.join(base_dir, f'{method}_on_surface.mat')
        free_save_path = os.path.join(base_dir, f'{method}_free_space.mat')

        ind = np.random.choice(pointcloud.points.shape[0], size=500000)
        savemat(surf_save_path, {
            'p': np.concatenate([pointcloud.points[ind, :], pointcloud.normals[ind, :]], axis=1)
        })
        savemat(free_save_path, {
            'p_sdf': sdfs_data
        })
    else:
        raise gr.Error('采样错误')


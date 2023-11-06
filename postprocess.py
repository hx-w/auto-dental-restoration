# -*- coding: utf-8 -*-

import numpy as np
import torch
from tqdm import tqdm
import trimesh
import skimage.measure
import seaborn as sns
import matplotlib.pyplot as plt

def extract_mesh(method, model, latent_code, resol: int, max_batch: int = 32 ** 3 * 4) -> trimesh.Trimesh:
    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-20/2, -20/2, -20/2]
    voxel_size = 20 / (resol - 1)

    overall_index = torch.arange(0, resol ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(resol ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % resol
    samples[:, 1] = (overall_index.long() / resol) % resol
    samples[:, 0] = ((overall_index.long() / resol) / resol) % resol

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = resol ** 3

    samples.requires_grad = False

    for head in tqdm(range(0, num_samples, max_batch), desc='正在提取零等值面网格'):
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            method.query_sdf(model, latent_code, sample_subset)
            .squeeze()
            .detach()
            .cpu()
        )

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(resol, resol, resol)

    mesh = build_mesh_from_sdf(sdf_values.data.cpu(), voxel_origin, voxel_size)

    ## post process
    res = sorted(
        mesh.split(only_watertight=False),
        key=lambda x: x.vertices.shape[0] * np.cumsum(x.bounding_box.primitive.extents)[-1],
        reverse=True
    )[0]
    res.remove_unreferenced_vertices()
    res.remove_degenerate_faces()
    res.remove_infinite_values()
    res.remove_duplicate_faces()
    res.vertex_normals

    return res

def build_mesh_from_sdf(pytorch_3d_sdf_tensor, voxel_grid_origin, voxel_size):
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0, spacing=[voxel_size] * 3
        )
    except Exception as e:
        raise f'MC error: {e}'

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    return trimesh.Trimesh(mesh_points, faces)

def create_slice_image(method, model, latent_code, resol, x_axis, y_axis, z_axis):
    sample_pnts = None
    if x_axis is not None:
        sample_pnts = torch.Tensor([
            (x_axis, (x / resol) * 20 - 20 / 2, (z / resol) * 20 - 20 / 2)
            for x in range(resol) for z in range(resol)
        ]).cuda()
    elif y_axis is not None:
        sample_pnts = torch.Tensor([
            ((x / resol) * 20 - 20 / 2, y_axis, (z / resol) * 20 - 20 / 2)
            for x in range(resol) for z in range(resol)
        ]).cuda()
    else:
        sample_pnts = torch.Tensor([
            ((x / resol) * 20 - 20 / 2, (z / resol) * 20 - 20 / 2, z_axis)
            for x in range(resol) for z in range(resol)
        ]).cuda()

    sdfs = (
        method.query_sdf(model, latent_code, sample_pnts)
        .squeeze(1).detach().cpu().numpy().reshape((resol, resol))
    )

    style = 'YlBuRd_r'
    plt.figure(figsize = (50, 50))
    htmap = sns.heatmap(sdfs, cmap=style, cbar=False, xticklabels=False, yticklabels=False)
    
    return htmap.get_figure()
    if filename:
        htmap.get_figure().savefig(filename, pad_inches=False, bbox_inches='tight')
# -*- coding: utf-8 -*-

import numpy as np
import torch
from tqdm import tqdm
import trimesh
import skimage.measure
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


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

def create_SDF_slice_from_mesh(pointcloud, resol: int, x_axis, y_axis, z_axis):
    sample_pnts = None
    if x_axis is not None:
        sample_pnts = torch.Tensor([
            (x_axis, (x / resol) * 20 - 20 / 2, (z / resol) * 20 - 20 / 2)
            for x in range(resol) for z in range(resol)
        ])
    elif y_axis is not None:
        sample_pnts = torch.Tensor([
            ((x / resol) * 20 - 20 / 2, y_axis, (z / resol) * 20 - 20 / 2)
            for x in range(resol) for z in range(resol)
        ])
    else:
        sample_pnts = torch.Tensor([
            ((x / resol) * 20 - 20 / 2, (z / resol) * 20 - 20 / 2, z_axis)
            for x in range(resol) for z in range(resol)
        ])

    sdfs = pointcloud.get_sdf_in_batches(sample_pnts, use_depth_buffer=False, sample_count=11)
    sdfs = np.clip(sdfs, -1.0, 1.0).reshape((resol, resol))

    style = 'coolwarm'
    plt.figure(figsize = (50, 50))
    htmap = sns.heatmap(sdfs, cmap=style, cbar=False, xticklabels=False, yticklabels=False)
    return htmap.get_figure()

def create_error_plot(pointcloud, ref_mesh):
    '''
    defect pointcloud to restored mesh
    '''
    errors = pointcloud.get_sdf_in_batches(ref_mesh.vertices, use_depth_buffer=False, sample_count=11)
    max_dist = np.max(errors)
    min_dist = np.min(errors)

    x_axis = np.linspace(min_dist, max_dist, 100)
    y_axis = np.zeros_like(x_axis)

    for i, x in tqdm(enumerate(x_axis), desc='计算误差分布'):
        y_axis[i] = len(errors[np.logical_and(errors >= x, errors < x + (max_dist - min_dist) / 100)])

    # df = pd.DataFrame({'ranges': x_axis, 'counts': y_axis})
    fig = px.line({'ranges': x_axis, 'counts': y_axis}, x='ranges', y='counts')
    fig.update_layout(
        title="误差分布图",
        xaxis_title="误差区间",
        yaxis_title="计数",
    )

    return fig
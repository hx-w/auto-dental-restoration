# -*- coding: utf-8 -*-

import os
import hashlib
from typing import List

import trimesh
import gradio as gr
from tqdm import tqdm
import torch

from method import VMethod
from preprocess import sample_sdfs
from postprocess import extract_mesh


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

class Context:
    def __init__(self, mesh_path: str, methods: List[VMethod], location: str, iter: int, resol: int, use_cache: bool=True):
        try:
            self.defect_mesh = trimesh.load(mesh_path)
            self.method_insts = methods
            self.location = location[1:] + '_Outside'
            self.iter = iter
            self.resol = resol
            self.cache = use_cache

            self.finished = False

            self.__prepare()
        except Exception as e:
            raise gr.Error(f'初始化错误: {e}')

    def __prepare(self):
        os.makedirs('.cache', exist_ok=True)

        # read mesh file and compute hash
        hash_inst = hashlib.md5()
        with open(self.defect_mesh.metadata['file_path'], 'rb') as f:
            while chunk := f.read(8192):
                hash_inst.update(chunk)
        
        self.tag = hash_inst.hexdigest()
        self.cache_dir = os.path.join('.cache', self.tag)
        os.makedirs(self.cache_dir, exist_ok=True)

    def preprocess(self, sample_ratios: List[int] = [1, 1, 1]):
        '''
        @param sample_ratios: [down, mid, up]
        '''
        mtds = []
        for mtd in tqdm(self.method_insts, desc='点云采样中'):
            if (
                self.cache
                and
                (
                    (f'{mtd}' == 'ToothDIT' and os.path.isfile(os.path.join(self.cache_dir, f'{mtd}_sdfs.npz')))
                    or
                    (
                        f'{mtd}' == 'DIF-Net' 
                        and os.path.isfile(os.path.join(self.cache_dir, f'{mtd}_on_surface.mat'))
                        and os.path.isfile(os.path.join(self.cache_dir, f'{mtd}_free_space.mat'))
                    )
                )
            ):
                gr.Info(f'命中缓存，跳过{mtd}采样环节')
                continue
            
            mtds.append(mtd.__str__())

        sample_sdfs(self.defect_mesh, mtds, self.cache_dir, sample_ratios)

    def reconstruct_latent(self):
        self.net_latent = {} # {'Method': ('model', 'latent')}
        for mtd in tqdm(self.method_insts, desc='正在应用所选模型'):
            skip = False
            latent_path = os.path.join(self.cache_dir, f'{mtd}_latent_{self.iter}.pth')
            if self.cache and os.path.isfile(latent_path):
                code = torch.load(latent_path)['latent_code']
                skip = True
                gr.Info(f'命中缓存，跳过{mtd}重建符号距离场环节')

            try:
                if skip:
                    _, model = mtd.inference(self.cache_dir, self.location, self.iter, skip=True)
                else:
                    code, model = mtd.inference(self.cache_dir, self.location, self.iter, skip=False)

                self.net_latent[f'{mtd}'] = (model, code)

                torch.save({'latent_code': code}, latent_path)
            except Exception as e:
                raise gr.Error(f'重建出错: {e}')


    def extract_surface(self):
        for mtd in tqdm(self.method_insts, desc='正在应用所选模型'):
            # cache built mesh
            recon_mesh_path = os.path.join(self.cache_dir, f'{mtd}_{self.iter}_{self.resol}_raw.obj')
            if self.cache and os.path.isfile(recon_mesh_path):
                gr.Info(f'命中缓存，跳过{mtd}提取零等值面网格环节')
                continue

            mesh = extract_mesh(mtd, *self.net_latent[f'{mtd}'], self.resol)
            mesh.export(recon_mesh_path)


    def post_compute(self, smooth: int):
        meshes, slices, errors = {}, {}, {}

        for mtd in tqdm(self.method_insts, desc='正在应用所选模型'):
            raw_mesh_path = os.path.join(self.cache_dir, f'{mtd}_{self.iter}_{self.resol}_raw.obj')
            final_mesh_path = os.path.join(self.cache_dir, f'{mtd}_{self.iter}_{self.resol}_final.obj')
            raw_mesh = trimesh.load(raw_mesh_path)
            # smooth with laplacian method
            final_mesh = trimesh.smoothing.filter_laplacian(raw_mesh, iterations=int(smooth))
            final_mesh.export(final_mesh_path)

            # create sdf slice
            ...

            # create error plot
            ...

            meshes[f'{mtd}'] = final_mesh_path
        
        self.finished = True

        return meshes, slices, errors

# -*- coding: utf-8 -*-

import os
import random
import json
import yaml
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.DIF.dataset import PointCloudMulti

class VMethod:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as ifh:
            self.config = yaml.safe_load(ifh)

        temp_list = os.listdir(self.config['templates_dir'])
        self.templates = {
            temp: os.path.join(self.config['templates_dir'], temp) for temp in temp_list
        }
        spec_list = os.listdir(self.config['specs_dir'])
        self.specs = {
            spec.split('.')[0]: os.path.join(self.config['specs_dir'], spec) for spec in spec_list
        }
        ckpt_list = os.listdir(self.config['ckpt_dir'])
        self.ckpts = {
            ckpt.split('.')[0]: os.path.join(self.config['ckpt_dir'], ckpt) for ckpt in ckpt_list
        }


    def __str__(self):
        return self.config['name']

    def inference(self, cache_dir: str, location: str, iter: int, skip: bool):
        '''
        return embedding latent code
        '''
        if location not in self.specs or not os.path.isfile(self.specs[location]):
            raise NotImplementedError(f'没有该牙位的模型: {location}')
        if location not in self.ckpts or not os.path.isfile(self.ckpts[location]):
            raise NotImplementedError(f'没有该牙位的模型: {location}')
        
        return self._inference_impl(cache_dir, location, iter, skip)

    def query_sdf(self, model, latent_code, queries):
        '''
        return sdf from queries
        '''
        raise NotImplementedError('没有实现')

class Mtd_ToothDIT(VMethod):
    def _inference_impl(self, cache_dir: str, location: str, iter: int, skip: bool):
        # load spec
        with open(self.specs[location], 'r') as jf:
            spec = json.load(jf)
        model_state = torch.load(self.ckpts[location])

        arch = __import__(
            (self.config['base_dir'] + '/model').replace('/', '.'),
            fromlist=[self.config['model_name']]
        )

        latent_size = spec['CodeLength']
        decoder = arch.Decoder(latent_size, **spec['NetworkSpecs'])
        decoder = torch.nn.DataParallel(decoder)
        decoder.load_state_dict(model_state['model_state_dict'])
        decoder = decoder.module.cuda()
        decoder.eval()

        if skip: return None, decoder

        data_sdf = self.__read_sdf_samples_into_ram(os.path.join(cache_dir, f'{self.config["name"]}_sdfs.npz'))

        data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

        ## Start Reconstruction
        def adjust_learning_rate(
            initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every, iter
        ):
            lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        decreased_by = 10
        stat = 0.01
        clamp_dist = 1.0
        num_samples = 8000
        lr = 5e-3
        l2reg = True

        adjust_lr_every = int(iter / 2)

        if type(stat) == type(0.1):
            latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
        else:
            latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

        latent.requires_grad = True

        optimizer = torch.optim.Adam([latent], lr=lr)

        loss_l1 = torch.nn.L1Loss()

        for e in tqdm(range(iter), desc='正在重建符号距离场'):
            decoder.eval()
            sdf_data = self.__unpack_sdf_samples_from_ram(
                data_sdf, num_samples
            )
            xyz = sdf_data['coords'].cuda()
            sdf_gt = sdf_data['sdfs'].cuda().unsqueeze(1)

            sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

            adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every, e)

            optimizer.zero_grad()

            latent_inputs = latent.expand(num_samples, -1)

            inputs = torch.cat([latent_inputs, xyz], 1).cuda().to(torch.float32)

            pred_sdf = decoder(inputs)

            if e == 0:
                pred_sdf = decoder(inputs)

            pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

            loss = loss_l1(pred_sdf, sdf_gt.reshape(pred_sdf.shape))
            if l2reg:
                loss += 1e-3 * torch.mean(latent.pow(2))
            loss.backward()
            optimizer.step()

        return latent, decoder
    
    def query_sdf(self, model, latent_code, queries):
        num_samples = queries.shape[0]
        latent_repeat = latent_code.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)
        with torch.no_grad():
            sdf = model(inputs)[:, :1]
        return sdf

    def __read_sdf_samples_into_ram(self, filename):
        npz = np.load(filename)
        pos_tensor = torch.from_numpy(npz["pos"])
        neg_tensor = torch.from_numpy(npz["neg"])
        surf_pnts_tensor = torch.from_numpy(npz["surf_pnts"])
        surf_norms_tensor = torch.from_numpy(npz["surf_norms"])
        on_surfs = torch.cat([surf_pnts_tensor, surf_norms_tensor], 1)

        return [pos_tensor, neg_tensor, on_surfs]

    def __unpack_sdf_samples_from_ram(self, data, subsample=None):
        if subsample is None:
            return data
        pos_tensor = data[0]
        neg_tensor = data[1]

        # split the sample into half
        half = int(subsample / 2)

        pos_size = pos_tensor.shape[0]
        neg_size = neg_tensor.shape[0]

        pos_start_ind = random.randint(0, pos_size - half)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

        if neg_size <= half:
            random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
            sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        else:
            neg_start_ind = random.randint(0, neg_size - half)
            sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

        samples = torch.cat([sample_pos, sample_neg], 0)
        randidx = torch.randperm(samples.shape[0])
        samples = torch.index_select(samples, 0, randidx)

        return {
            'coords': samples[:, :3],
            'sdfs': samples[:, 3:]
        }


class Mtd_DIFNet(VMethod):
    def _inference_impl(self, cache_dir: str, location: str, iter: int, skip: bool):
        with open(self.specs[location], 'r') as jf:
            spec = yaml.safe_load(jf)
        model_state = torch.load(self.ckpts[location])

        arch = __import__(
            (self.config['base_dir'] + '/model').replace('/', '.'),
            fromlist=[self.config['model_name']]
        )

        model = arch.DeformedImplicitField(**spec)
        model.load_state_dict(model_state)

        # The network should be fixed for evaluation.
        for param in model.template_field.parameters():
            param.requires_grad = False
        for param in model.hyper_net.parameters():
            param.requires_grad = False

        model.cuda()
        model.eval()

        if skip: return None, model

        sdf_dataset = PointCloudMulti(
            paths=[os.path.join(cache_dir, f'{self.config["name"]}_{tail}.mat') for tail in ['on_surface', 'free_space']],
            **spec
        )
        dataloader = DataLoader(sdf_dataset, shuffle=True, collate_fn=sdf_dataset.collate_fn, batch_size=1, num_workers=0, drop_last=True)
        
        # start train
        embedding = model.latent_codes(torch.zeros(1).long().cuda()).clone().detach() # initialization for evaluation stage
        embedding.requires_grad = True

        optim = torch.optim.Adam(lr=spec['lr'], params=[embedding])

        len_ds = len(dataloader)
        epochs = max(10, iter // len_ds)

        for epoch in tqdm(range(epochs), desc='正在重建符号距离场'):
            for model_input, gt in dataloader:
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                losses = model.embedding(embedding, model_input, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    train_loss += single_loss

                optim.zero_grad()
                train_loss.backward()
                optim.step()

        return embedding, model

    def query_sdf(self, model, latent_code, queries):
        queries = queries[None, ...]
        return model.inference(queries, latent_code)

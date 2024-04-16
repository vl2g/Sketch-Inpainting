# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image

from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import get_model_parameters_info, instantiate_from_config
from image_synthesis.data.build import build_dataloader
import image_synthesis.data.sketch_inpainting_mscoco as sk_inp_ds
from torch.utils.data import DataLoader

class VQ_Diffusion():
    def __init__(self, config, path):
        self.info = self.get_model(ema=True, model_path=path, config_path=config)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad=False

    def get_model(self, ema, model_path, config_path):
        if 'OUTPUT' in model_path: # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else: 
            model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)
        model = build_model(config)
        model_parameters = get_model_parameters_info(model)
        
        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")

        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']
        elif 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 0

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)

        if ema==True and 'ema' in ckpt:
            print("Evaluate EMA model")
            ema_model = model.get_ema_model()
            missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)
        
        return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

    def inference_generate_sample_with_class(self, text, truncation_rate, save_root, batch_size,fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['label'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+'r',
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.jpg')
            im = Image.fromarray(content[b])
            im.save(save_path)

    def inference_generate_sample_with_condition(self, text, truncation_rate, save_root, batch_size,fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['text'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)

    def inference_generate_sample_with_sketch(self, batch, truncation_rate, save_root, fast=False, epoch=0):
        os.makedirs(save_root, exist_ok=True)

        str_cond = ''
        save_root_ = os.path.join(save_root, epoch)
        os.makedirs(save_root_, exist_ok=True)

        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=batch,
                filter_ratio=0,
                replicate=1,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        # save results
        content = model_out['content']
        sketch = batch['sketch'].permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        obj_mask = batch['obj_mask'].to('cpu').unsqueeze(-1).repeat(1, 1, 1, 3).numpy()
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        blended = ((1. - obj_mask) * (batch['image'].permute(0, 2, 3, 1).to('cpu').numpy()) + obj_mask * content).astype(np.uint8)
        masked_img = ((1. - obj_mask) * content + obj_mask).astype(np.uint8)

        for b in range(content.shape[0]):
            cnt = b
            save_base_name = batch['sketch_file'][b].replace('_out.png', '')
            print(f'Saving {save_base_name}')
            save_path = os.path.join(save_root_, save_base_name+'_inpainted.jpg')
            im = Image.fromarray(content[b])
            im.save(save_path)

            Image.fromarray(blended[b]).save(save_path.replace('_inpainted', '_blended'))

            Image.fromarray(sketch[b]).save(os.path.join(save_root_, save_base_name+'_sketch.png'))
            Image.fromarray(masked_img[b]).save(os.path.join(save_root_, save_base_name+'_masked.png'))


if __name__ == '__main__':
    # VQ_Diffusion = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/human_pretrained.pth')
    # VQ_Diffusion.inference_generate_sample_with_condition("a man with beard",truncation_rate=0.86, save_root="RESULT",batch_size=2,fast=2)  # fast is a int from 2 to 10
    # VQ_Diffusion.inference_generate_sample_with_condition("a beautiful smiling woman",truncation_rate=0.85, save_root="RESULT",batch_size=8)

    base_pth = './OUTPUT/sketch_inp_diff_edges'

    # for original_config: use config.orig
    config_pth = os.path.join(base_pth, 'configs', 'config.yaml')
    config = load_yaml_config(config_pth)
    config['dataloader']['validation_datasets'][0]['target'] = 'image_synthesis.data.sketch_inpainting_mscoco.COCOSketchInpaintDatasetQSketchList'

    print(config['dataloader'])

    # for no lpips - required no more
    # print(config['model'].keys())
    # config['model']['params']['diffusion_config']['lpips_net'] = None

    val_ds = sk_inp_ds.COCOSketchInpaintDatasetQSketchList(
        data_root='../data/',
        phase='validation',
        sketch_subdir='obj_edge_maps',
        sketch_list_file='/DATA/nakul/sketch/data/validation/sketch_list_all.txt'
    )
    print('-'*100)
    print(len(val_ds))

    # val_dl = build_dataloader(config)['validation_loader']
    val_dl = DataLoader(val_ds, batch_size=150, pin_memory=True, num_workers=16)

    VQ_Diffusion = VQ_Diffusion(config=config_pth, path='OUTPUT/sketch_inp_diff_edges/checkpoint/last.pth')


    for i, batch in enumerate(val_dl):
        for k, v in batch.items():
            if type(v) == torch.Tensor:
                batch[k] = v.cuda()

        VQ_Diffusion.inference_generate_sample_with_sketch(batch=batch, truncation_rate=0.85, save_root="RESULT", epoch='final_inf/all_edges_297_085')
 

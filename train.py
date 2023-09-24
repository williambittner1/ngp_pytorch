from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch import nn
import cv2
import numpy as np
from opt import get_opts

from losses import NeRFLoss

from models.networks import NGP

from models.rendering import render, MAX_SAMPLES

from utils import load_ckpt, slim_ckpt

from torch.utils.data import DataLoader

from datasets import dataset_dict

from datasets.ray_utils import axisangle_to_R, get_rays

from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from kornia.utils.grid import create_meshgrid3d

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.plugins import DDPPlugin

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualSimilarity


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        
        super().__init__()

        self.save_hyperparameters(hparams) 

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = "None" if hparams.use_exposure else "Sigmoid"
        self.model = NGP(scale=hparams.scale, rgb_act=rgb_act)
        G = self.model.grid_size
        self.model.register_buffer('density_grid', torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords', create_meshgrid3d(G, G, G, normalized_coordinates=False, dtype=torch.int32).reshape(-1, 3))


    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['poses']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3:] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {  'test_time': split!='train',
                    'random_bg': self.hparams.random_bg}
        
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, **kwargs)
        

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy
        self.test_dataset = dataset(split='test', **kwargs)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR', nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT', nn.Parameter(torch.zeros(N, 3, device=self.device)))
        
        load_ckpt(self.model, self.hparams.weight_path) # load pretrained checkpoint (excluding optimizers, etc.)

        net_params = []
        for name, param in self.named_parameters():
            if name not in ['dR', 'dT']:
                net_params.append(param)

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)]   # learning rate is hard-coded here
        net_sch = CosineAnnealingLR(self.net_opt, self.hparams.num_epochs, self.hparams.lr/30)  # network scheduler

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=None,
                          num_workers=16,
                          persistent_workers=True,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)
     
    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step % self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                            warmup=self.global_step<self.warmup_steps,
                                            erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)

        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance, 
                                            **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = 0.5 * (unit_exposure_rgb - self.train_dataset.unit_exposure_rgb)**2

        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad:
            self.train_psnr(results['rgb'], batch['rgb'])
        
        self.log('lr', self.net_opt.param_groups[0]['lr']) 
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/raymarching_samples', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/volume_rendering_samples', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss
    








if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('Please specify a @ckpt_path (checkpoint path) for validation!')
    
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirPath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)  
    
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f'logs/{hparams.dataset_name}',
                               name=hparams.exp_name,
                               default_hp_metric=False)
    
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only:
        ckpt_ = slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                          save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

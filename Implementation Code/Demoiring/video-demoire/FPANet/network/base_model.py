# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR), NAFNet(https://github.com/megvii-research/NAFNet) 
# Copyright 2018-2020 BasicSR Authors
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import logging
import os
from network import define_network
from utils.utils import get_root_logger, imwrite, tensor2img, spatial_to_frequency
from utils.dist_util import get_dist_info
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils.dist_util import master_only
import datetime
RUN_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
loss_module = importlib.import_module('losses.losses')
metric_module = importlib.import_module('metric.metric')

logger = logging.getLogger('FPANet')

class Model():
    """Base Model Class"""
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
            
        if train_opt.get('frequency_opt'):
            percep_type = train_opt['frequency_opt'].pop('type')
            cri_frequency_cls = getattr(loss_module, percep_type)
            self.cri_frequency = cri_frequency_cls(
                **train_opt['frequency_opt']).to(self.device)
        else:
            self.cri_frequency = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        optimizer_type = train_opt['optim_g'].pop('type')
        print('..',  optimizer_type)
        self.optimizer_g = torch.optim.AdamW([{'params': optim_params}], **train_opt['optim_g'])
            
        
    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        print('..',  scheduler_type)
        self.schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_g, **train_opt['scheduler'])
        

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if len(data) != 4:
            self.next = data['next'].to(self.device)
            self.prev = data['prev'].to(self.device)
        if 'gt' in data:
            self.gt_tmp = data['gt'].to(self.device)
            self.gt = [self.gt_tmp]
            self.gt.append(F.interpolate(self.gt_tmp, scale_factor=0.5, mode='bilinear'))
            self.gt.append(F.interpolate(self.gt_tmp, scale_factor=0.25, mode='bilinear'))


    def optimize_parameters(self):
        self.optimizer_g.zero_grad()
        # Forward process
        if self.opt['datasets']['train']['mode'] == 'multi':
            data = [self.lq, self.prev, self.next]
            preds = self.net_g(data)
        else:
            data = [self.lq]
            preds = self.net_g(data)
            
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds

        # Calculate loss
        l_total = 0
        loss_dict = OrderedDict()
        
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            
            l_pix += self.cri_pix(self.output[0], self.gt[0])
            l_pix += self.cri_pix(self.output[1], self.gt[1])
            l_pix += self.cri_pix(self.output[2], self.gt[2])

            l_total += l_pix
            loss_dict['l_pix'] = l_pix


        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output[0], self.gt[0])
            l_percep_2, l_style = self.cri_perceptual(self.output[1], self.gt[1])
            l_percep_4, l_style = self.cri_perceptual(self.output[2], self.gt[2])
            if l_percep is not None:
                l_total += l_percep 
                l_total += l_percep_2 
                l_total += l_percep_4 
                loss_dict['l_percep'] = l_percep + l_percep_2 + l_percep_4
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        
        if self.cri_frequency:
            l_mag_pix = 0.
            l_pha_pix = 0.
            # Frequency L1 loss
            out_mag, out_pha = spatial_to_frequency(self.output[0])
            gt_mag, gt_pha = spatial_to_frequency(self.gt[0])

            l_mag_pix += self.cri_frequency(out_mag, gt_mag)
            l_pha_pix += self.cri_frequency(out_pha, gt_pha)

            ### size 1/2
            out_mag, out_pha = spatial_to_frequency(self.output[1])
            gt_mag, gt_pha = spatial_to_frequency(self.gt[1])

            l_mag_pix += self.cri_frequency(out_mag, gt_mag)
            l_pha_pix += self.cri_frequency(out_pha, gt_pha)
            ### size 1/4
            out_mag, out_pha = spatial_to_frequency(self.output[2])
            gt_mag, gt_pha = spatial_to_frequency(self.gt[2])

            l_mag_pix += self.cri_frequency(out_mag, gt_mag)
            l_pha_pix += self.cri_frequency(out_pha, gt_pha)
            
            l_total += (l_mag_pix + l_pha_pix)
            
            loss_dict['l_mag_pix'] = l_mag_pix
            loss_dict['l_phapix'] = l_pha_pix


        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                if self.opt['datasets']['val']['mode'] == 'multi':
                    data = [self.lq[i:j], self.prev[i:j], self.next[i:j]]
                else:
                    data = [self.lq[i:j]]
                pred = self.net_g(data)
                pred = pred[0]
                # if isinstance(pred, list):
                #     pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, save_img):
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        processed_idx = 0  # add this before the loop

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            save_img_path = osp.join(self.opt['path']['visualization'], 'test', f'{img_name}.jpg')

            if os.path.exists(save_img_path):
                print(f"[SKIP] Already exists: {save_img_path}")
                continue

            print(f"[DEBUG] Processing {processed_idx}: {val_data['lq_path'][0]}")
            processed_idx += 1

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=True)

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=True)
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                # Create a subfolder using run ID
                run_folder = osp.join(self.opt['path']['visualization'], 'test', RUN_ID)
                os.makedirs(run_folder, exist_ok=True)

                save_img_path = osp.join(run_folder, f'{img_name}.jpg')
                save_gt_img_path = osp.join(run_folder, f'{img_name}_gt.jpg')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            # calculate metrics
            opt_metric = deepcopy(self.opt['val']['metrics'])
            for name, opt_ in opt_metric.items():
                metric_type = opt_.pop('type')
                self.metric_results[name] += getattr(
                    metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            collected_metrics = OrderedDict()
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)
            self.collected_metrics = collected_metrics

            keys = []
            metrics = []
            for name, value in self.collected_metrics.items():
                keys.append(name)
                metrics.append(value)
            metrics = torch.stack(metrics, 0)

            if self.opt['dist']:
                torch.distributed.reduce(metrics, dst=0)

            if self.opt['rank'] == 0:
                metrics_dict = {}
                cnt = 0
                for key, metric in zip(keys, metrics):
                    if key == 'cnt':
                        cnt = float(metric)
                        continue
                    metrics_dict[key] = float(metric)
                for key in metrics_dict:
                    metrics_dict[key] /= cnt
                self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'], metrics_dict)

        return 0.


    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        for i, o in enumerate(resume_optimizers):
            self.optimizer_g.load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers.load_state_dict(s)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            self.schedulers.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)
            
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt[0].detach().cpu()
        return out_dict

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        print(' load net keys', load_net.keys)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)


    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
        
    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict


    def validation(self, dataloader, current_iter, save_img=False):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            return self.dist_validation(dataloader, current_iter, save_img)
        # else:
        #     pass
        else:
            return self.nondist_validation(dataloader, current_iter,
                                    save_img)
    
    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters',
                                                  False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net
    
    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net
    
    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            cnt = 0
            for v in sorted(list(crt_net_keys)):
                cnt += 1
                logger.warning(f'  {v}')
            logger.warning(cnt)
            logger.warning('Loaded net - current net:')
            cnt = 0
            for v in sorted(list(load_net_keys )):
                cnt += 1
                logger.warning(f'  {v}')
            logger.warning(cnt)

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizers': [],
                'schedulers': []
            }
            state['optimizers'].append(self.optimizer_g.state_dict())
            state['schedulers'].append(self.schedulers.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'],
                                     save_filename)
            torch.save(state, save_path)

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        print(save_path)
        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)
        
    def get_current_learning_rate(self):
        return [
            param_group['lr']
            for param_group in self.optimizer_g.param_groups
        ]
        
    def get_current_log(self):
        return self.log_dict
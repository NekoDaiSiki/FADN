import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
# import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, Gradient_Loss, SSIM_Loss
import numpy as np
import torch.nn.functional as F
import math
from models.modules.Subnet_constructor import subnet


logger = logging.getLogger('base')

class FADN(BaseModel):
    def __init__(self, opt):
        super(FADN, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = define_G(opt).to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()

        self.load()

        if self.is_train:
            self.netG.train()
            # self.hyper.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.Rec_Forw_grad = Gradient_Loss()
            self.Rec_back_grad = Gradient_Loss()
            self.Rec_forw_SSIM = SSIM_Loss()
            self.Rec_back_SSIM = SSIM_Loss()

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))


                        
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  
        self.real_H = data['GT'].to(self.device)  
        self.noisy_H = data['Noisy'].to(self.device)  

    def feed_test_data(self, data):
        self.noisy_H = data.to(self.device)  

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        # l_forw_grad = 0.1* self.train_opt['lambda_fit_forw'] * self.Rec_Forw_grad(out, y)
        # l_forw_SSIM = self.train_opt['lambda_fit_forw'] * self.Rec_forw_SSIM(out, y).mean()

        return l_forw_fit # + l_forw_grad + l_forw_SSIM

    def loss_backward(self, x,  y,gau, yL_list,yH_list):
        loss_z0 = self.train_opt['lamda_loss_z0'] * self.Reconstruction_back(y[0],yL_list[0])
        loss_z1 = self.train_opt['lamda_loss_z1'] * self.Reconstruction_back(y[1],yL_list[1])

        x_samples,high_fre = self.netG(x=y,gaussian_input = gau, rev=True)
        x_samples_image = x_samples[0]


        loss_h1 = self.train_opt['lamda_loss_h1'] * torch.mean(torch.abs(high_fre[1] - yH_list[1]))
        loss_h0 = self.train_opt['lamda_loss_h0'] * torch.mean(torch.abs(high_fre[0] - yH_list[0]))

        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)
        l_grad_back_rec = 0.1*self.train_opt['lambda_rec_back_grid'] * self.Rec_back_grad(x, x_samples_image)
        l_back_SSIM = self.train_opt['lambda_rec_back_SSIM'] * self.Rec_back_SSIM(x, x_samples_image).mean()
        return l_back_rec+ l_grad_back_rec + l_back_SSIM , loss_z0, loss_z1, loss_h0,loss_h1
       #  return l_back_rec


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # forward
        self.output,gaussian,yL_list, yH_list = self.netG(x=self.noisy_H,y=self.real_H)

        l_back_rec,loss_z0,loss_z1,loss_h0,loss_h1 = self.loss_backward(self.real_H, self.output,gaussian,yL_list,yH_list)

        # total loss
        loss = loss_z0 + l_back_rec + loss_z1 + loss_h0 + loss_h1
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['loss_z0'] = loss_z0.item()
        self.log_dict['loss_z1'] = loss_z1.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['loss_h0'] = loss_h0.item()
        self.log_dict['loss_h1'] = loss_h1.item()
        self.log_dict['loss'] = loss.item()

    def test(self, self_ensemble=False):
        self.input = self.noisy_H

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        # self.hyper.eval()

        with torch.no_grad():
            if self_ensemble:
                forward_function = self.netG.forward
                self.fake_H = self.forward_x8(self.input, forward_function, gaussian_scale)
            else:
                output, gaussian, _,_ = self.netG(x=self.input)
                self.forw_L = output[-1]
                fake_H,high_fre = self.netG(x=output, gaussian_input=gaussian ,rev=True)
                self.fake_H = fake_H[0][:, :3, :, :]


        self.netG.train()
        # self.hyper.train()

    def test_time(self,*args):
        self.input = self.noisy_H
        self.netG.eval()
        output, gaussian, _ = self.netG(x=self.input)
        self.forw_L = output[-1]

        y_forw = torch.cat((output[-1], gaussian[-1]), dim=1)
        output[-1] = y_forw
        fake_H,high_fre = self.netG(x=output, gaussian_input=gaussian ,rev=True)
        self.fake_H = fake_H[0]

    def MC_test(self, sample_num=16, self_ensemble=False):
        self.input = self.noisy_H

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        with torch.no_grad():
            if self_ensemble:
                forward_function = self.netG.forward
                self.fake_H = self.Multi_forward_x8(self.input, forward_function, gaussian_scale, sample_num)
            else:
                output = self.netG(x=self.input)
                self.forw_L = output[:, :3, :, :]
                fake_Hs = []
                for i in range(sample_num):
                    y_forw = torch.cat((output[-1][:, :3, :, :], gaussian_scale * self.gaussian_batch(output[-1][:, 3:, :, :].shape)), dim=1)
                    fake_Hs.append(self.netG(x=y_forw, rev=True)[:, :3, :, :])
                fake_H = torch.cat(fake_Hs, dim=0)
                self.fake_H = fake_H.mean(dim=0, keepdim=True)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['Denoised'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['Noisy'] = self.noisy_H.detach()[0].float().cpu()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def forward_x8(self, x, forward_function, gaussian_scale):
        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            return ret

        noise_list = [x]
        for tf in 'v', 'h', 't':
            noise_list.extend([_transform(t, tf) for t in noise_list])

        lr_list = [forward_function(x=aug) for aug in noise_list]

        sr_list = [forward_function(x=data[0],gaussian_input=data[1], rev=True)[0][0] for data in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


    def Multi_forward_x8(self, x, forward_function, gaussian_scale, sample_num=16):
        
        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            return ret

        noise_list = [x]
        for tf in 'v', 'h', 't':
            noise_list.extend([_transform(t, tf) for t in noise_list])

        lr_list = [forward_function(aug) for aug in noise_list]
        sr_list = []
        for data in lr_list:
            fake_Hs = []
            for i in range(sample_num):
                y_forw = torch.cat((data[:, :3, :, :], gaussian_scale * self.gaussian_batch(data[:, 3:, :, :].shape)), dim=1)
                fake_Hs.append(self.netG(x=y_forw, rev=True)[:, :3, :, :])
            fake_H = torch.cat(fake_Hs, dim=0)
            fake_H = fake_H.mean(dim=0, keepdim=True)
            sr_list.append(fake_H)

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    netG = InvNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num)

    return netG


class InvNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[8,8], down_num=2):
        super(InvNet, self).__init__()
        self.downnum = down_num
        self.backbone = [] 
        self.low_freqs = []
        self.high_freqs = []

        current_channel = channel_in
        for i in range(down_num):
            subG_tmp = []
            if i != 0:
                current_channel = current_channel -current_channel//4
            
            b = HaarDownsampling(current_channel)
            subG_tmp.append(b)
            current_channel = (current_channel)*4
            
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, current_channel//4)
                subG_tmp.append(b)
            subG_tmp = nn.ModuleList(subG_tmp) 
            self.backbone.append(subG_tmp)

            self.low_freqs.append(Enhance(current_channel//4,current_channel//4))
            self.high_freqs.append(Restoration(current_channel-current_channel//4,current_channel-current_channel//4))


        self.backbone = nn.ModuleList(self.backbone)
        self.low_freqs = nn.ModuleList(self.low_freqs)
        self.high_freqs = nn.ModuleList(self.high_freqs)

    def forward(self, x, y = None,gaussian_input = [],rev=False, cal_jacobian=False):
        out = x
        result_low = []
        result_res = []
        if not rev:
            yl_list = []
            yh_list = []
            if  y!=None:
                y_out = y 
                for i in range(self.downnum): 
                    for op in self.backbone[i]: 
                        y_out = op.forward(y_out, rev)
                    channel = y_out.size(1)
                    y_save = y_out
                    yl_list.append(y_save[:,:channel//4,:,:])      
                    yh_list.append(y_save[:,channel//4:,:,:])      
                    y_out = y_out[:,channel//4:,:,:]


            for i in range(self.downnum): 
                for op in self.backbone[i]:
                    out = op.forward(out, rev)

                channel = out.size(1)
                out_low = out[:,:channel//4,:,:]
                out_high = out[:,channel//4:,:,:]
                dn_low = self.low_freqs[i](out_low)
                result_low.append(dn_low)
                dn_high = self.high_freqs[i](out_high)
                result_res.append(dn_high)

                out = out_high

            
            return result_low, result_res ,yl_list, yh_list# 
        
        else:
            result_list = []
            high_frequency = []
            high_frequency.append(gaussian_input[-1])

            for i in range(self.downnum-1,-1,-1): 
                if i == self.downnum-1:
                    res = torch.cat([out[i],gaussian_input[i]],dim=1)
                else:
                    res = res + 0.25*gaussian_input[i]
                    high_frequency.append(res)
                    res = torch.cat((out[i],res),dim=1)
                
                for op in reversed(self.backbone[i]):
                    res = op.forward(res, rev)
                    if cal_jacobian:
                        jacobian += op.jacobian(out, rev)
                result_list.append(res)
            result_list.reverse()
            high_frequency.reverse()
            return result_list,high_frequency




class Enhance(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Enhance, self).__init__()
        feature = 32
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.conv2(residual)
        out = x + residual
        return out

class Restoration(nn.Module):
    def __init__(self,channel_in,channel_out):
        super().__init__()
        self.down_dim = nn.Conv2d(channel_in,channel_in//4,1)
        self.up_dim = nn.Conv2d(channel_in//4*2,channel_out,1)
        self.atten = AttentionBlock(channel_out)
        self.conv_list = []
        for i in range(5):
            self.conv_list.append(nn.Conv2d(channel_out,channel_out,3,padding=1))
            self.conv_list.append(nn.LeakyReLU())
        self.conv_list = nn.Sequential(*self.conv_list)
        self.conv_se = nn.Conv2d(channel_out,channel_out,3,padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel_out,channel_out,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_out,channel_out,bias=False),
            nn.Sigmoid()       
         )
    def forward(self,x):
        b,c,h,w = x.shape
        x = self.down_dim(x)
        x = self.up_dim(torch.cat([x,torch.randn_like(x)],dim=1))
        x = self.atten(x)
        x = self.conv_list(x)
        x = self.conv_se(x)
        y = self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        x = y*x + x
        return x


class AttentionBlock(nn.Module):
    def __init__(self,N=192):
        super().__init__()
        self.conv = nn.Conv2d(N,N,kernel_size=3,padding=1)
    def forward(self,x):
        atten = torch.sigmoid(self.conv(F.leaky_relu(x)))
        x = x*atten + x
        return x
    

def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)




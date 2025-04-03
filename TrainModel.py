import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from ImageDataset import ImageSeqDataset, ImageSeqDatasetEval, ImageSeqDatasetMultiEval
from batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution, \
    TensorRandom, TensorRGBToYCbCr
from loss.PerceptualLoss import PerceptualLoss
from loss.SSIMLoss import SSIMLoss
from loss.LumiLoss import BrightnessOrderLoss


from metrics import compute_psnr_ssim, psnr_ssim_for_folder, psnr_ssim
from sec.co_lut_arch import CoNet

EPS = 1e-8


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        '''
        self.train_rgb_transform = transforms.Compose([
            BatchRandomResolution(config.high_size, interpolation=2),
            BatchToTensor()
        ])
        '''
        self.train_rgb_transform = transforms.Compose([
            BatchTestResolution(2048, interpolation=2),
            BatchToTensor()
        ])

        self.test_rgb_transform = transforms.Compose([
            BatchTestResolution(2048, interpolation=2),
            BatchToTensor()
        ])
        self.train_batch_size = 1
        self.test_batch_size = 1
        # training set configuration
        self.train_data = ImageSeqDataset(csv_file=os.path.join(config.trainset, 'train.txt'),
                                          hr_img_seq_dir=config.trainset,
                                          rgb_transform=self.train_rgb_transform)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=1)

        # testing set configuration
        self.test_data = ImageSeqDatasetEval(csv_file=os.path.join(config.testset, 'test.txt'),
                                             hr_img_seq_dir=config.testset,
                                             gt_img_dir=config.testGTset,
                                             rgb_transform=self.test_rgb_transform)

        self.test_multi_data = ImageSeqDatasetMultiEval(csv_file=os.path.join(config.testset, 'test.txt'),
                                                        hr_img_seq_dir=config.testset,
                                                        gt_img_dirs=["expert_a_testing_set",
                                                                     "expert_b_testing_set",
                                                                     "expert_c_testing_set",
                                                                     "expert_d_testing_set",
                                                                     "expert_e_testing_set"],
                                                        rgb_transform=self.test_rgb_transform)

        self.test_loader = DataLoader(self.test_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)

        # initialize the model
        self.secModel = CoNet()
        self.secModel_name = type(self.secModel).__name__
        print(self.secModel)

        # loss function
        self.perceptual_loss_fn = PerceptualLoss()
        self.ssim_loss_fn = SSIMLoss()
        self.lumi_loss_fn = BrightnessOrderLoss()
        
        # self.lumi_loss_fn = BrightnessOrderLoss(num_channels=8)
        # self.lumi_struc_loss_fn = BrightnessStructurePreservationLoss(num_channels=8)
        # optimizer
        self.initial_lr = config.lr
        self.optimizer_sec = optim.Adam(
            self.secModel.parameters(),
            lr=self.initial_lr,
            betas=(0.9, 0.99),
            weight_decay=0
        )

        # we don't want to use multiple gpus, because it is going to split
        # the sequence into multiple sub-sequences
        # if torch.cuda.device_count() > 1 and config.use_cuda:
        #     print("[*] GPU #", torch.cuda.device_count())
        #     self.mefModel = nn.DataParallel(self.mefModel)

        if torch.cuda.is_available() and config.use_cuda:
            self.secModel.cuda()
            self.perceptual_loss_fn = self.perceptual_loss_fn.cuda()
            self.ssim_loss_fn = self.ssim_loss_fn.cuda()
            self.lumi_loss_fn = self.lumi_loss_fn.cuda()

        # some states
        self.epochs_warmup = config.epochs_warmup - 1
        self.epochs_stable = config.epochs_stable - 1
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.max_results = {'psnr_sec': 0.0, 'ssim_sec': 0.0, 'psnr_mef': 0.0, 'ssim_mef': 0.0}
        self.ckpt_path = config.ckpt_path
        self.use_cuda = config.use_cuda
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save
        self.fused_img_path = config.fused_img_path
        self.corrected_img_path = config.corrected_img_path
        self.weight_map_path = config.weight_map_path
        self.eval_save = config.eval_save
        self.eval_per_img = config.eval_per_img
        self.gt_path = config.testGTset
        # try load the model
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt_path, config.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)

        def lr_lambda_sec(epoch):
            # Warm-up阶段（0-9）：线性下降 1.0 → 0.1
            if epoch <= self.epochs_warmup:
                return 1.0 - 0.9 * (epoch / self.epochs_warmup)
            # 联合阶段（10-29）：余弦退火 1.0 → 0.0
            elif epoch <= self.epochs_stable:
                return 1.0
            else:
                joint_epoch = epoch - self.epochs_stable
                total_joint = self.max_epochs - self.epochs_stable
                return 0.5 * (1 + math.cos(math.pi * joint_epoch / total_joint))

        self.scheduler_sec = lr_scheduler.LambdaLR(self.optimizer_sec, lr_lambda=lr_lambda_sec)

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch(epoch)

    def get_alpha(self, epoch):
        '''
                if epoch <= self.epochs_stable:
            joint_epoch = (epoch - self.epochs_warmup) // 1
            total_joint = (self.epochs_stable - self.epochs_warmup) // 1
            alpha = joint_epoch / total_joint
            if alpha > 0.5:
                alpha = 0.5
            return alpha
        '''
        if epoch <= self.epochs_stable:
            return 0.5
        else:
            return 0.5

    def mertens_fusion(self, i_hr):
        i_hr_np = i_hr.numpy()
        i_hr_np = (i_hr_np * 255).astype(np.uint8)
        rgb_images = []
        for img in i_hr_np:
            img = img.transpose((1, 2, 0))
            rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            rgb_images.append(rgb_img)
        mertensFusion = cv2.createMergeMertens()
        fused_img = mertensFusion.process(rgb_images) * 255
        fused_img = cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB)
        fused_img = fused_img.transpose((2, 0, 1))
        fused_img = torch.from_numpy(fused_img / 255).float().cuda()
        fused_img = fused_img.unsqueeze(0)
        return fused_img

    def transfer_intermediate(self, i_rgb, i_gt=None, case=None, cases=None, epoch=None, state=None):
        temp_rgb = i_rgb.detach()
        temp_rgb = temp_rgb.cuda()
        with torch.no_grad():
            temp_sec, _, _ = self.secModel(temp_rgb)
            temp_sec = temp_sec.detach().cpu()

        if state == 'eval':
            save_sec = temp_sec.detach()
            i_gt = i_gt.expand_as(save_sec)
            sum_psnr, sum_ssim = compute_psnr_ssim(save_sec, i_gt)
            if self.eval_save:
                epoch_save_path = os.path.join(self.corrected_img_path, f"epoch{epoch}")
                save_path = os.path.join(epoch_save_path, case[0])
                # 创建保存路径
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for i, img_tensor in enumerate(save_sec):
                    # 获取对应的图片名称
                    img_name, _ = os.path.splitext(cases[i][0])
                    img_tensor = img_tensor.unsqueeze(0)
                    self._save_image(img_tensor, save_path, img_name)

        if state == 'train':
            return temp_sec
        else:
            return temp_sec, sum_psnr, sum_ssim

    def _train_single_epoch(self, epoch):

        # start training
        print('SEC learning rate: {:f}'.format(self.optimizer_sec.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):

            i_rgb = sample_batched['I_rgb']
            i_rgb = torch.squeeze(i_rgb, dim=0)
            if epoch > self.epochs_warmup:
                i_hr = self.transfer_intermediate(i_rgb, epoch=epoch, state='train')
                i_sec = torch.cat([i_hr, i_rgb], dim=0)
                O_hr_RGB = self.mertens_fusion(i_sec)
                self._save_image(O_hr_RGB, self.weight_map_path, f"{epoch}_{step}_mixed")
            else:
                O_hr_RGB = self.mertens_fusion(i_rgb)
            # SECModel
            self.optimizer_sec.zero_grad()
            self.secModel.train()
            i_rgb = i_rgb.cuda()
            O_sec, lumi_feature, lut_feature = self.secModel(i_rgb)
            O_hr_RGB = O_hr_RGB.expand_as(O_sec)
            l_pixel = nn.L1Loss()(O_sec, O_hr_RGB)
            l_perceptual = self.perceptual_loss_fn(O_sec, O_hr_RGB)
            l_ssim = self.ssim_loss_fn(O_sec, O_hr_RGB)
            l_lumi = self.lumi_loss_fn(lumi_feature)
            secLoss = l_pixel + 0.1*l_perceptual + l_lumi +0.5*l_ssim
            secLoss.backward()
            self.optimizer_sec.step()
            format_str = 'SEC: (E:%d, S:%d) [Loss = %.4f]'
            print(format_str % (epoch, step, secLoss.data.item()))

        self.scheduler_sec.step()

        if (epoch + 1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            avg_psnr_sec, avg_ssim_sec, avg_psnr_mef, avg_ssim_mef = self.eval(epoch)
            #self.test_results.append(test_results)
            out_str_mef = 'MEF: Epoch {} Testing: PSNR: {:.4f} SSIM: {:.4f}'.format(epoch,
                                                                                    avg_psnr_mef,
                                                                                    avg_ssim_mef)
            out_str_sec = 'SEC: Epoch {} Testing: PSNR: {:.4f} SSIM: {:.4f}'.format(epoch,
                                                                                    avg_psnr_sec,
                                                                                    avg_ssim_sec)
            with open(self.corrected_img_path + "/res_log.txt", 'a') as f:
                f.write(out_str_sec + "\n")
            with open(self.fused_img_path + "/res_log.txt", 'a') as f:
                f.write(out_str_mef + "\n")
            print(out_str_mef)
            print(out_str_sec)
            if avg_psnr_sec > self.max_results['psnr_sec'] or avg_ssim_sec > self.max_results['ssim_sec']:
                self.max_results['psnr_sec'] = avg_psnr_sec
                self.max_results['ssim_sec'] = avg_ssim_sec
                model_name = 'sec-best-epoch{:0>4d}.pt'.format(epoch)
                model_name = os.path.join(self.ckpt_path, model_name)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.secModel.state_dict(),
                    'optimizer': self.optimizer_sec.state_dict(),
                    'test_psnr': avg_psnr_sec,
                    'test_ssim': avg_ssim_sec
                }, model_name)
            if avg_psnr_mef > self.max_results['psnr_mef'] or avg_ssim_mef > self.max_results['ssim_mef']:
                self.max_results['psnr_mef'] = avg_psnr_mef
                self.max_results['ssim_mef'] = avg_ssim_mef
                model_name = 'mef-bestsec-epoch{:0>4d}.pt'.format(epoch)
                model_name = os.path.join(self.ckpt_path, model_name)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.secModel.state_dict(),
                    'optimizer': self.optimizer_sec.state_dict(),
                    'test_psnr': avg_psnr_mef,
                    'test_ssim': avg_ssim_mef
                }, model_name)

        return [secLoss.data.item()]

    def eval(self, epoch):
        scores = []
        psnr_sec = []
        ssim_sec = []
        psnr_mef = []
        ssim_mef = []
        for step, sample_batched in enumerate(self.test_loader, 0):
            # TODO: remove this after debugging
            i_rgb, i_gt, cases, case = (sample_batched['I_rgb'],
                                        sample_batched['I_gt'],
                                        sample_batched['cases'],
                                        sample_batched['case'])
            i_rgb = torch.squeeze(i_rgb, dim=0)
            i_gt = torch.squeeze(i_gt, dim=0)
            i_hr, psnr_values, ssim_values = self.transfer_intermediate(i_rgb, i_gt, case, cases, epoch, state='eval')
            psnr_sec.extend(psnr_values)
            ssim_sec.extend(ssim_values)
            '''
            i_sec = torch.cat([i_hr, i_rgb], dim=0)
            O_hr_RGB = self.mertens_fusion(i_sec).cpu()
            '''
            O_hr_RGB = 0.5*self.mertens_fusion(i_hr).cpu() + 0.5*self.mertens_fusion(i_rgb).cpu()
            if self.eval_save:
                epoch_save_path = os.path.join(self.fused_img_path, f"epoch{epoch}")
                if not os.path.exists(epoch_save_path):
                    os.makedirs(epoch_save_path)
                img_name = case[0]
                self._save_image(O_hr_RGB, epoch_save_path, img_name)
            psnr_values_mef, ssim_values_mef = compute_psnr_ssim(O_hr_RGB, i_gt)
            psnr_mef.extend(psnr_values_mef)
            ssim_mef.extend(ssim_values_mef)
            # self._save_image(W_hr, self.weight_map_path, str(epoch) + '_' + str(step))
        avg_psnr_sec = sum(psnr_sec) / len(psnr_sec)
        avg_ssim_sec = sum(ssim_sec) / len(ssim_sec)
        avg_psnr_mef = sum(psnr_mef) / len(psnr_mef)
        avg_ssim_mef = sum(ssim_mef) / len(ssim_mef)
        if self.eval_save and self.eval_per_img:
            psnr_ssim(os.path.join(self.fused_img_path, f"epoch{epoch}"), self.gt_path)
            psnr_ssim_for_folder(os.path.join(self.corrected_img_path, f"epoch{epoch}"), self.gt_path)
        return avg_psnr_sec, avg_ssim_sec, avg_psnr_mef, avg_ssim_mef

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
        torch.save(state, filename)

    def _save_image(self, image, path, name):
        b = image.size()[0]
        for i in range(b):
            t = image.data[i]
            t[t > 1] = 1
            t[t < 0] = 0
            utils.save_image(t, "%s/%s.jpg" % (path, name))

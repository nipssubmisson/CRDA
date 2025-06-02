# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from torch.distributions import Categorical
from sklearn.decomposition import PCA
from pytorch_ssim import ssim





parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml',
                    help='path to detector YAML file')
parser.add_argument("--detector_path2", type=str,)
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)

global domain_dim
domian_dim = 7

def calculate_ssim(images1, images2):
    # 确保输入张量在同一个设备上
    if images1.device != images2.device:
        images2 = images2.to(images1.device)

    # 确保window_size是奇数，padding是整数
    window_size = 11  # 必须是奇数
    padding = window_size // 2  # 整数除法
    
    # 创建高斯窗口
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    # 获取通道数
    channel = images1.size(1)
    window = create_window(window_size, channel).to(images1.device)

    # 计算SSIM
    mu1 = F.conv2d(images1, window, padding=(padding, padding), groups=channel)
    mu2 = F.conv2d(images2, window, padding=(padding, padding), groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(images1*images1, window, padding=(padding, padding), groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(images2*images2, window, padding=(padding, padding), groups=channel) - mu2_sq
    sigma12 = F.conv2d(images1*images2, window, padding=(padding, padding), groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

class PPOBuffer:
    def __init__(self, batch_size=16):
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.values = []
        self.batch_size = batch_size

    def store(self, state, action, old_log_prob, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.old_log_probs.append(old_log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def get_batches(self):
        # 将数据转换为 Tensor
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.old_log_probs)
        rewards = torch.tensor(self.rewards)
        values = torch.tensor(self.values)
        # 清空缓冲池
        self.states, self.actions, self.old_log_probs, self.rewards, self.values = [], [], [], [], []
        return states, actions, old_log_probs, rewards, values
    def clear(self):
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.values = []

def compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95):
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    advantages = []
    advantage = 0
    for delta in reversed(deltas):
        advantage = delta + gamma * gae_lambda * advantage
        advantages.insert(0, advantage)
    # 指定 advantages 张量的设备与 values 相同
    advantages = torch.tensor(advantages, device=values.device)
    returns = advantages + values[:-1]
    return advantages, returns

def ppo_loss(new_log_probs, old_log_probs, advantages, values, returns, clip_epsilon=0.2):
    # 重要性权重
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    # 裁剪损失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 价值函数损失
    value_loss = F.mse_loss(values, returns)
    
    return policy_loss + 0.5 * value_loss


# def apply_pca(real_features: torch.Tensor, target_dimension: int = 512) -> torch.Tensor:
#     # 确保张量在CPU上
#     real_features_cpu = real_features.cpu()
    
#     # 将real_features转换成适合PCA处理的形状，即[n_samples, n_features]
#     real_features_flatten = real_features_cpu.view(real_features_cpu.size(0), -1).numpy()
    
#     # 初始化PCA，并设置需要降到的目标维度
#     pca = PCA(n_components=target_dimension)
#     # 应用PCA
#     real_features_reduced = pca.fit_transform(real_features_flatten)
    
#     # 将结果转换回Tensor并放回原始设备（这里假设原始设备可能是GPU）
#     original_device = real_features.device
#     real_features_reduced_tensor = torch.tensor(real_features_reduced, dtype=real_features.dtype, device=original_device)
#     return real_features_reduced_tensor
import math

class DCTTransform(nn.Module):
    def __init__(self):
        super(DCTTransform, self).__init__()
        self.dct_kernel = self._create_dct_kernel()

    def _create_dct_kernel(self, N=8):
        # 创建DCT核
        kernel = torch.zeros(N, N)
        for k in range(N):
            kernel[k, :] = torch.cos((torch.arange(0, N) + 0.5) * (k * 3.141592653589793 / N))
        kernel[0, :] /= torch.sqrt(torch.tensor(2.0))
        return kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)  # 将通道维度合并到batch维度
        
        # 计算padding以保持输出尺寸不变
        pad = self.dct_kernel.size(-1) // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        
        x = F.conv2d(x, self.dct_kernel.to(x.device), stride=1, padding=0)
        
        # 计算新的H和W
        new_H = x.size(2)
        new_W = x.size(3)
        x = x.view(B, C, new_H, new_W)  # 恢复原始形状
        return x
def comp_entropy(masks):
    # masks: [N, C, H, W]
    probs = masks.mean(dim=0)  
    log_probs = torch.log(probs + 1e-10)  
    return -(probs * log_probs).sum() 

###########我在这里进行组件的准备
###首先是mask的生成
from AdvXAIDeepfakes.explanation.evaluation.adversarial_image_generation import generate_mask_batch

###伪造方法引擎，我这里想要用现有的数据集替代 先不写460192
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(4, 3, 3),
            padding=(0, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(4, 3, 3),
            padding=(0, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        # 如果输入通道数和输出通道数不同，需要使用1x1卷积调整通道数
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
###策略网络
class PolicyNet(nn.Module):
    def __init__(self, action_dim=domian_dim-1):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Conv3d(
                in_channels=2048,
                out_channels=512,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=512,
                out_channels=256,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )
        
        self.value_net = nn.Sequential(
            nn.Conv3d(
                in_channels=2048,
                out_channels=512,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=512,
                out_channels=256,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        
    def get_action(self, state):
        # 原有逻辑不变，但需要返回旧策略的 log_prob
        logits = self.policy_net(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        print(logits.argmax(dim=1))
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return logits, log_prob, entropy

    # def evaluate_actions(self, state, action):
    #     # 评估新策略的 log_prob 和熵
    #     logits = self.policy_net(state)
    #     dist = Categorical(logits=logits)
    #     new_log_prob = dist.log_prob(action)
    #     entropy = dist.entropy().mean()
    #     return new_log_prob, entropy
    def evaluate_actions(self, state, action_probs):
        # Evaluate the log_prob and entropy of the new policy
        logits = self.policy_net(state)
        dist = Categorical(logits=logits)

        batch_size, num_actions = action_probs.size()

        actions = action_probs.argmax(dim=-1)  # 或者使用torch.multinomial(action_probs, 1).squeeze(-1)进行采样

        # 现在actions的形状为(batch_size,)，符合log_prob的要求
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()  # 直接计算策略分布的熵并取平均
        return log_probs, entropy
                
    def get_value(self, state):
        return self.value_net(state)

###adversarial blend 控制blend的强度
class AdversarialBlender(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_net = nn.Sequential(
            nn.Conv2d(4,32,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,1,3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x_real, x_fakes, mask, policy_action):
        # 转换输入维度
        x_fakes_tensor = torch.stack(x_fakes, dim=1)  # [B, D, C, H, W]
        
        # 概率加权混合

        tau = 1.0  # 温度参数，可调整
        hard = not self.training  # 训练时soft，推理时hard
        
        # 直接使用 policy_action 作为 logits
        gumbel_policy = F.gumbel_softmax(
            policy_action,  # 直接传入 logits（policy_action）
            tau=tau,
            hard=hard,
            dim=1  # 沿着D维度（动作维度）进行softmax
        )
        
        # 计算加权混合
        main_method = torch.einsum('bd,bdchw->bchw', gumbel_policy, x_fakes_tensor)
    
        blended = x_real * (1 - mask) + main_method * mask
        
        # Alpha生成（强制0.9-1.0范围）
        alpha_input = torch.cat([blended, mask], dim=1)
        base_alpha = self.alpha_net(alpha_input) 
        alpha = base_alpha * 0.1 + 0.9  # 线性映射到[0.9, 1.0]
        
        # 最终输出
        output = x_real * (1 - alpha) + blended * alpha

        return output
      
####最后的生成器



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_and_value_net = PolicyNet()
        self.adversarial_blender = AdversarialBlender()
    def compute_feature_diversity(self, x, detector):
        """计算batch内样本特征差异度"""
        features = detector.features({'image': x, 'label': torch.zeros(len(x), device=x.device)})
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze()
        
        # 标准化特征向量
        normalized_feat = F.normalize(features, p=2, dim=1)  # [B, D]
        
        # 计算余弦相似度矩阵
        cosine_sim = torch.mm(normalized_feat, normalized_feat.T)  # [B, B]
        
        # 计算差异度量（排除对角线）
        mask = torch.triu(torch.ones_like(cosine_sim), diagonal=1).bool()
        diversity = (1 - cosine_sim[mask]).mean() * 2  # 缩放至[0,2]范围
        
        return diversity
    def forward(self, x_real,x_fakes,detector,segments_area):
        mask = generate_mask_batch(model=detector,images=x_real,target_class=1,number_selected_segments=segments_area)
        device = next(detector.parameters()).device
        self.policy_and_value_net = self.policy_and_value_net.to(device)
        self.adversarial_blender = self.adversarial_blender.to(device)
        x_real = x_real.to(device)
        x_fakes = [fake.to(device) for fake in x_fakes]
        mask = torch.from_numpy(mask).float().to(device)
        with torch.no_grad():
        
            x_real_dict = {'image': x_real, 'label': torch.zeros(x_real.size(0), dtype=torch.long).to(device)}
            
            real_features = detector.features(x_real_dict)
            fake_features = [detector.features({'image': fake, 'label': torch.zeros(fake.size(0), dtype=torch.long).to(device)}) for fake in x_fakes]
            # conv1_weight = self.adversarial_blender.alpha_net[0].weight.to(device)  
            # conv2_weight = self.adversarial_blender.alpha_net[2].weight.to(device)  
            
            # real_features = real_features.view(real_features.size(0), -1)
            # fake_features = torch.cat([f.view(f.size(0), -1) for f in fake_features], dim=1)
            # blender_params = torch.cat([conv1_weight.view(-1), conv2_weight.view(-1)]).unsqueeze(0).repeat(real_features.size(0), 1)  # 添加batch维度
            # state = torch.cat([real_features, fake_features, blender_params], dim=1)
            fake_features = torch.stack(fake_features, dim=1)
            real_features = real_features.unsqueeze(1)
            # 假设fake_features形状为(3, 64, 64)
            # 假设real_features形状为(64, 64)
            # print(fake_features.shape)

            state = torch.cat([fake_features, real_features], dim=1)

            state = state.permute(0, 2, 1, 3, 4)

        action, log_prob,entropy = self.policy_and_value_net.get_action(state)


        x_fake_adv= self.adversarial_blender(x_real,x_fakes,mask,action)
        value = self.policy_and_value_net.get_value(state)
        # x_fake_adv_dict = {'image': x_fake_adv, 'label': torch.ones(x_fake_adv.size(0), dtype=torch.long).to(device)}
        # diversity = self.compute_feature_diversity(x_fake_adv,detector)
        return x_fake_adv,mask,action,log_prob,value,entropy,state




def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])



def prepare_training_data(config):
    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda' or config['model_name'] == 'optlsda':
            train_set = LSDADataset(config, mode='train')

        else:
            print(config['model_name'])
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'sbiplus_v2':
        train_set = SBIPlusV2Dataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] =='ppo':
        train_set = SBIPlusPPODataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='train',
                )
        
    if config['model_name'] == 'lsda' or config['model_name'] == 'optlsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler, 
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                )
    return train_data_loader


def prepare_valid_data(config):
    valid_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
            )
    valid_data_loader = \
        torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=valid_set.collate_fn,
            drop_last = True,
        )
    return valid_data_loader




    

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
        else:
            test_set = LRLDataset(
                config=config,
                mode='test',
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last = (test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders



def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    with open(args.detector_path2, 'r') as f:
        config_fake = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2_fake = yaml.safe_load(f)
    if 'label_dict' in config_fake:
        config2_fake['label_dict']=config_fake['label_dict']
    config_fake.update(config2_fake)
    config_fake['local_rank']=args.local_rank
    if config_fake['dry_run']:
        config_fake['nEpochs'] = 0
        config_fake['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config_fake['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config_fake['test_dataset'] = args.test_dataset
    config_fake['save_ckpt'] = args.save_ckpt
    config_fake['save_feat'] = args.save_feat

    config_fake['lmdb'] = False
    if config_fake['lmdb']:
        config_fake['dataset_json_folder'] = 'preprocessing/dataset_json_v3'

    config_fake['ddp']= args.ddp
    init_seed(config_fake)

    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat


    config['lmdb'] = False
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    for key, value in config_fake.items():
        params_string += "fake    {}: {}".format(key, value) + "\n"
    logger.info(params_string)
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    # print(config["train_bacthSize"])
    twotype_train_data_loader = prepare_training_data(config)
    # fivetype_train_set = SBIPlusV2Dataset(config, mode='train')

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # valid_data_loader = prepare_valid_data(config)

    # if 'dataset_adv' in config:
    #     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    train_data_loader = prepare_training_data(config_fake)



    # prepare the model (detector)
    ######我想要借用prodet的多种fake的数据集，但是我又需要普通的训练方式,就用sbi训练方式来预热
    # config['model_name'] = 'sbi'

    model_class = DETECTOR[config['model_name']]
    model = model_class(config)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)
    
    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    # start training
    #######这里首先需要预热几个epoch,暂定为5，之后在开始dynamic training
    ppo_buffer = PPOBuffer()
    clip_epsilon = 0.2  # PPO裁剪参数
    ppo_epochs = 3     # PPO优化轮数
    generater = Generator()
    optim_g = optim.Adam([
        {'params': generater.policy_and_value_net.policy_net.parameters()},
        {'params': generater.policy_and_value_net.value_net.parameters()},
        {'params': generater.adversarial_blender.alpha_net.parameters()},
    ], lr= 1e-3,weight_decay=1e-4)
    mask_history = []


    env_high_entropy = []   # 高熵环境 (E_adv1)
    env_med_entropy = []    # 中熵环境 (E_adv2)
    env_low_entropy = []     # 低熵环境 (E_adv3)
    env_capacity = 16       # 每个环境的最大样本数
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        if epoch < 0:
            if epoch != 7:
                best_metric = trainer.train_epoch(                  
                            epoch=epoch,
                            train_data_loader=twotype_train_data_loader,
                            test_data_loaders=None,)
            else:
                best_metric = trainer.train_epoch(                  
                            epoch=epoch,
                            train_data_loader=twotype_train_data_loader,
                            test_data_loaders=test_data_loaders,)
        else:
            ####我们需要根据train_data_loader来修改twotype_train_data_loader,首先更新一轮
            # for i,data_dict in enumerate(train_data_loader):
            #     generater.train()
            generater.train()
            ppo_buffer.clear()

            for i,data_dict in tqdm(enumerate(train_data_loader)):
                batch_size = int(data_dict['image'].shape[0]/domian_dim)
                x_fakes = data_dict['image'][batch_size:]
                # x_fakes = x_fakes.view(3,4,3,256,256)
                x_fake_labels = data_dict['label'][batch_size:]
                x_fakes_list = []
                for i in range(domian_dim-1):
                    x_fakes_list.append(x_fakes[batch_size*i:batch_size*i+batch_size])
                x_real = data_dict['image'][:batch_size]
        
                x_fake, mask, action,old_log_prob, value,entropy,state = generater(x_real,x_fakes_list,detector=model,segments_area=50-epoch)

                x_real = x_real.to(x_fake.device)

                
                assert x_fake.shape == x_real.shape
                
                # if fake_adv_features.dim() > 2:
                #     fake_adv_features = 

                # predictions = model(data_dict)
                # score = model.get_losses(data_dict, predictions)["overall"]
                # with torch.no_grad():
                if 1:

                    feature_diversity = generater.compute_feature_diversity(x_fake,model)
                    x_fake_dict = {'image': x_fake, 'label': torch.ones(x_fake.size(0), dtype=torch.long).to(x_fake.device)}

                    pred = model(x_fake_dict)["cls"]
                    # attack_success= (pred.argmax(dim=1) == 1).float().mean()
                    attack_success = pred[:, 1].sigmoid().mean()
                    from skimage.metrics import structural_similarity as SSIM
                    # ssim_teacher = SSIM(window_size=11,channel=3)                  
                    # ssim_score = ssim_teacher(x_fake,x_real)
                    ssim_score = calculate_ssim(x_fake,x_real)
                    sparse_penalty = mask.mean()
                    mask_history.append(mask)
                    if len(mask_history)>100:
                        mask_history.pop(0)
                    hist_entropy = comp_entropy(torch.cat(mask_history,dim=0))
                    
                    DCT_extractor = DCTTransform()
                    real_dct = DCT_extractor(x_real)
                    fake_dct = DCT_extractor(x_fake)
                    fake_device = fake_dct.device
                    real_dct = real_dct.to(fake_device)
                    freq_kl = F.kl_div(fake_dct.log(),real_dct,reduction='batchmean')
                    reward = 0.4*(1-attack_success)+ 0.1*ssim_score  + 0.05*hist_entropy + 0.45*feature_diversity  #+ 0.1 *sparse_penalty
                    # reward = 0.*(1-attack_success)+ 0*ssim_score  + 0*hist_entropy + 0.6*feature_diversity

                    
                    # reward = torch.clamp(reward, min=0.0, max=1.0)
                ppo_buffer.store(
                    state=state,  # 根据实际状态定义调整
                    action=action, 
                    old_log_prob=old_log_prob.detach(),  # 必须detach
                    reward=reward,
                    value=value.detach().mean()
                )


                advantage = reward - value.detach()   
                policy_loss = - (old_log_prob*advantage).mean()
                value_loss = F.mse_loss(value,reward)
                print("policy_loss:",policy_loss)
                if torch.isnan(value_loss).any():
                    print("Warning: value_loss is NaN")
                    value_loss = torch.zeros_like(value_loss)
                entropy_loss = - hist_entropy * 0.05
            
                total_loss = policy_loss + 0.2* value_loss + entropy_loss - 0.05*entropy

                # before_params = [param.clone() for param in generater.policy_and_value_net.parameters()]
                # optim_g.zero_grad()
                # reward.backward()
                # has_grad = False
                # for name, param in generater.policy_and_value_net.named_parameters():
                #     if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                #         has_grad = True
                #         print(f"Gradient for {name}")
                #     else:
                #         print(f"No gradient for {name}")
                # if has_grad:
                #     print("Gradient backpropagation occurred.")
                # else:
                #     print("No gradient backpropagation occurred.")
                # optim_g.step()

                trainer.optimizer.zero_grad()
                images = torch.cat([x_real, x_fake], dim=0)  # 在 batch 维度拼接
                labels = torch.cat([
                    torch.zeros(len(x_real), dtype=torch.long, device=x_real.device),  # 真实图像标签为0
                    torch.ones(len(x_fake), dtype=torch.long, device=x_fake.device)    # 生成图像标签为1
                ], dim=0)
                images = images.clone().detach().requires_grad_(True)

                
                images_len = images.shape[0]
                domain_images_len = x_fakes.shape[0]

                x_fakes = x_fakes.to(images.device)
                x_fake_labels = x_fake_labels.to(dtype=torch.long, device=labels.device) - 1
                new_images = torch.cat((images, x_fakes), dim=0)
                new_labels = torch.cat((labels, x_fake_labels), dim=0)


                sorted_entropy, _ = torch.sort(entropy_per_sample)
                low_threshold = sorted_entropy[int(0.33 * len(sorted_entropy))]
                high_threshold = sorted_entropy[int(0.66 * len(sorted_entropy))]
                
                # 将样本分配到不同环境
                for j in range(x_fake.size(0)):
                    entropy_val = entropy_per_sample[j].item()
                    sample = x_fake[j].detach().clone()
                    
                    if entropy_val > high_threshold:
                        env_high_entropy.append(sample)
                        if len(env_high_entropy) > env_capacity:
                            env_high_entropy.pop(0)  # FIFO
                    elif entropy_val > low_threshold:
                        env_med_entropy.append(sample)
                        if len(env_med_entropy) > env_capacity:
                            env_med_entropy.pop(0)
                    else:
                        env_low_entropy.append(sample)
                        if len(env_low_entropy) > env_capacity:
                            env_low_entropy.pop(0)

                env_high_entropy = torch.stack(env_high_entropy)
                env_med_entropy = torch.stack(env_med_entropy)
                env_low_entropy = torch.stack(env_low_entropy)
                env_high_label = torch.ones(env_high_entropy.size(0), dtype=torch.long, device=labels.device)
                env_med_label = torch.ones(env_med_entropy.size(0), dtype=torch.long, device=labels.device)
                env_low_label = torch.ones(env_low_entropy.size(0), dtype=torch.long, device=labels.device)
                causal_images = torch.cat((new_images, env_high_entropy, env_med_entropy, env_low_entropy), dim=0)
                causal_labels = torch.cat((new_labels, env_high_label, env_med_label, env_low_label), dim=0)


                new_data_dict = {"image": new_images, 
                                "label": new_labels,
                                "images_len":images_len,
                                "domain_images_len":domain_images_len,
                                "causal_images":causal_images,
                                "causal_labels":causal_labels}
                
                
                trainer.train_step(new_data_dict)
                


               
                if len(ppo_buffer.rewards) >= 5:  # 确保有足够数据
                    
                    print("policy network updating")

                    # print(ppo_buffer.rewards)
                    # print(ppo_buffer.values)
                    # 转换数据为Tensor
                    states = torch.stack(ppo_buffer.states).cuda()
                    actions = torch.stack(ppo_buffer.actions).cuda()
                    old_log_probs = torch.stack(ppo_buffer.old_log_probs).cuda()
                    rewards = torch.tensor(ppo_buffer.rewards).cuda()
                    values = torch.tensor(ppo_buffer.values).cuda()

                    # 计算GAE
                    advantages, returns = compute_gae(rewards, values)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 多epoch优化
                    for _ in range(ppo_epochs):
                        # 随机打乱数据
                        indices = torch.randperm(len(advantages))
                        for idx in indices:
                            batch_states = states[idx]
                            batch_actions = actions[idx]
                            batch_old_log_probs = old_log_probs[idx]
                            batch_advantages = advantages[idx]
                            batch_returns = returns[idx]
                    
                            # 计算新策略概率
                            new_log_probs, entropy = generater.policy_and_value_net.evaluate_actions(
                                batch_states, 
                                batch_actions
                            )
                            
                            # 重要性采样比率
                            ratio = torch.exp(new_log_probs - batch_old_log_probs)
                            
                            # PPO裁剪损失
                            surr1 = ratio * batch_advantages
                            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                            policy_loss = -torch.min(surr1, surr2).mean()

                            value_pred = generater.policy_and_value_net.get_value(batch_states)
                            value_loss = F.mse_loss(value_pred, batch_returns)

                            entropy_loss = -entropy.mean()
                            total_loss = policy_loss + 0.2 * value_loss + 0.01 * entropy_loss
                            # print("total_loss:",total_loss)
                            optim_g.zero_grad()
                            total_loss.backward()
                            # for name, param in generater.named_parameters():
                            #     if param.grad is not None:
                            #         grad_norm = torch.norm(param.grad).item()
                            #         print(f"Gradient norm for {name}: {grad_norm}")
                            #     else:
                            #         print(f"No gradient for {name}")

                            # torch.nn.utils.clip_grad_norm_(generater.parameters(), 0.5)
                            optim_g.step()

                    ppo_buffer.clear()
                    torch.cuda.empty_cache()

            best_metric = trainer.test_epoch(
                epoch=epoch,
                iteration=10,
                test_data_loaders=test_data_loaders,
                step = 1000,
            )
 
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()

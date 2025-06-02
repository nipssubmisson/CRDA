
import os
import datetime
import logging
import random
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='CRDA')
class CoreDetectorDomain(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_domains = config.get('num_domains', 6)  # 假设默认6个域
        self.backbone = self.build_backbone(config)
        self.domain_classifier = self.build_domain_classifier()  # 新增域分类器
        self.domain_loss_weight = config.get('domain_loss_weight', 0.3)  # 域分类损失权重
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # 原有代码保持不变
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        state_dict = torch.load(config['pretrained'])
        # ...权重处理...
        backbone.load_state_dict(state_dict, False)
        return backbone

    def build_causal_classifier(self):
        return  nn.Sequential(
            # 输入通道调整为 2048，与输入特征的通道数匹配
            nn.Conv2d(
                in_channels=2048,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            # 自适应平均池化层，将特征图尺寸调整为 (1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def build_domain_classifier(self):
        # 新增域分类器构建
        return  nn.Sequential(
            # 输入通道调整为 2048，与输入特征的通道数匹配
            nn.Conv2d(
                in_channels=2048,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            # 自适应平均池化层，将特征图尺寸调整为 (1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, self.num_domains),
        )

        
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        if "images_len" in data_dict.keys() and "causal_images" in data_dict.keys():
            label = data_dict['label']
            image_len = data_dict["images_len"]
            label_detect = data_dict['label'][:image_len]
            label_domain = data_dict['label'][image_len:]

            pred_detect = pred_dict['pred_detect']

            core_feat_detect = pred_dict['core_feat_detect']
            loss_detect = self.loss_func(core_feat_detect, pred_detect, label_detect)
            
            causal_images = data_dict['causal_images']
            causal_labels = data_dict['causal_labels']
            causal_pred = self.causal_classifier(causal_images)
            causal_loss = self.loss_func(causal_pred, causal_labels)

            loss_domain = torch.nn.functional.cross_entropy(pred_dict['prob_domain'], label_domain)
            loss_all = loss_detect + loss_domain + causal_loss
            loss_dict = {'overall': loss_all,
                        'loss_detect': loss_detect,
                        'loss_domain': loss_domain,
                        'loss_causal': causal_loss}
            return loss_dict
        else:
            label = data_dict['label']
            pred = pred_dict['cls']
            core_feat = pred_dict['core_feat']
            loss = self.loss_func(core_feat, pred, label)
            loss_dict = {'overall': loss}
            return loss_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # 原有特征提取
        features = self.features(data_dict)
        if "images_len" in data_dict.keys():
            images_len = data_dict["images_len"]
            domain_images_len = data_dict["domain_images_len"]
            features_detect = features[:images_len]
            features_domain = features[images_len:]

            core_feat_detect = nn.ReLU(inplace=False)(features_detect)
            core_feat_detect= F.adaptive_avg_pool2d(core_feat_detect, (1, 1))
            core_feat_detect = core_feat_detect.view(core_feat_detect.size(0), -1)
            pred_detect = self.classifier(features_detect)
            prob_detect = torch.softmax(pred_detect, dim=1)[:, 1]
            
            # 域分类分支

            pred_domain = self.domain_classifier(features_domain)
            prob_domain = F.softmax(pred_domain, dim=1)
            
            # 添加域预测输出
            pred_dict = {
                "pred_detect": pred_detect,
                "prob_detect": prob_detect,
                "pred_domain": pred_domain,
                "prob_domain": prob_domain,
                "feature_detect": features_detect,
                "core_feat_detect": core_feat_detect,
                "feature_domain": features_domain,
            }
        else:
            core_feat = nn.ReLU(inplace=False)(features)
            core_feat= F.adaptive_avg_pool2d(core_feat, (1, 1))
            core_feat = core_feat.view(core_feat.size(0), -1)
            # get the prediction by classifier
            pred = self.classifier(features)
            # get the probability of the pred
            prob = torch.softmax(pred, dim=1)[:, 1]
            # build the prediction dict for each output
            pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'core_feat': core_feat}

        return pred_dict
    

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    def features(self, data_dict: dict) -> torch.tensor:
        # get the features from backbone

        return self.backbone.features(data_dict['image'])
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

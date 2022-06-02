import os
import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
import numpy as np
from utils import get_optimizer, setup_logger
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc

from models import AGDA

def initModel(mod, gpu_ids):
    mod = mod.to(f'cuda:{gpu_ids[0]}')
    mod = nn.DataParallel(mod, gpu_ids)
    return mod

class Trainer(): 
    def __init__(self, gpu_ids, model, optimizer_choice, lr, max_epoch):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.model = model
        self.model = initModel(self.model, gpu_ids)
        self.optimizer = get_optimizer(optimizer_choice, self.model, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=5,
                                                         gamma=0.5)
        self.epoch = 0
        self.max_epoch = max_epoch
        self.weight = [1, 0.1, 1, 0.1]
        self.AG = AGDA(
            kernel_size=11, dilation=2, sigma=7, threshold=(0.4, 0.6),
            zoom=(3, 5), scale_factor=0.5, noise_rate=0.1, mode='soft'
        ) if self.weight[2] != 0 else None


    def forward(self, x):
        out = self.model(x)

        return out

    def train_loss(self, loss_pack):
        if 'loss' in loss_pack:
            return loss_pack['loss']
        loss = \
            self.weight[0] * loss_pack['ensemble_loss'] \
            + self.weight[1] * loss_pack['aux_loss']
        if self.weight[2] != 0:
            loss += \
                self.weight[2] * loss_pack['AGDA_ensemble_loss'] \
                + self.weight[3] * loss_pack['match_loss']
        return loss

    def optimize_weight(self, x, label):

        x, label = x.to(self.device), label.to(self.device)
        with torch.set_grad_enabled(True):
            loss_pack = self.model(x, label, AG=self.AG)

        loss = self.train_loss(loss_pack)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def epoch_step(self, decay_rate):
        if hasattr(self.model.module, "auxiliary_loss"):
            self.model.module.auxiliary_loss.alpha *= decay_rate
        self.scheduler.step()
        self.epoch += 1

    def goon(self):
        return (self.epoch < self.max_epoch)

def evaluate(model, dataset_img):
    bz = 32
    # torch.cache.empty_cache()
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset = d,
                batch_size = bz,
                shuffle = True,
                num_workers = 8
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                img = img.detach().cuda()
                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())




    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true,y_pred,pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true==0)[0]
    idx_fake = np.where(y_true==1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)

    return AUC, r_acc, f_acc
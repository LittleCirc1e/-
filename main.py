import os
import sys
import time
import random
import numpy as np
import argparse

import torch
import torch.nn

from trainer import Trainer, evaluate
from dataset import get_dataset, FFDataset
from models import MAT_v2
from utils import setup_logger, path_provider

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('-bz', type=int, default=12, help="batch size")
parser.add_argument('-e', type=int, default=0, help="epoch number")
parser.add_argument('-am', type=str, default='normal', help="attention module mode")
parser.add_argument('-dm', type=str, default='original', help="data mode")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
gpu_ids = [*range(osenvs)]

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)



if __name__ == "__main__":
    frame_size = 160
    frame_num = 20
    batch_size = args.bz
    data_path, test_path, ckpt_path = path_provider(args.dm, "distance")
    loss_freq = 100
    alpha_decay = 0.95

    # load train set
    dataset = FFDataset(
        dataset_root=os.path.join(data_path, 'train', 'real'), size=frame_size,
        frame_num=frame_num, augment=True)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8)

    dataset_img, total_len = get_dataset(name='train', size=frame_size,
                                         root=test_path, frame_num=50,
                                         augment=True)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8
    )
    len_dataloader = dataloader_real.__len__()


    root = data_path
    mode = 'valid'
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    dataset_real_te = FFDataset(dataset_root=real_root, size=frame_size, frame_num=frame_num, augment=False)
    dataset_fake_te, _ = get_dataset(name=mode, root=origin_root, size=frame_size, frame_num=frame_num, augment=False)
    dataset_img_test = torch.utils.data.ConcatDataset([dataset_real_te, dataset_fake_te])

    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    model = MAT_v2(
        back_bone='xception', num_class=1, pretrained='imagenet', input_size=(frame_size, frame_size),
        attention_layers=["b7","b9"], atte_module_chocie=args.am, M=8,
        feature_layer='b1', feature_choice='v2', DCT_choice=2,
        loss_choice='v2', margins=(0.5, [0.1, -2]), alpha=0.05,
        final_dim=256, cb_mode='fuse', rg_dropout_rate=0.25, fn_dropout_rate=0.5
    )
    ck = torch.load("logs/saved_ck/7_auc_0.971792355371901_racc_0.865909090909091_facc_0.9472727272727273.pth")
    new_ck = {
        key[7:]:ck[key] for key in ck.keys()
    }
    model.load_state_dict(new_ck)
    model.freeze([])
    trainer = Trainer(gpu_ids, model, optimizer_choice="SGD", lr=args.lr, max_epoch=args.e)
    trainer.total_steps = 0

    highest_auc = 0
    while trainer.goon():
        epoch = trainer.epoch
        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)

        logger.debug(f'No {epoch}')
        i = 0

        s_time = time.time()
        while i < len_dataloader:
            trainer.total_steps += 1

            try:
                data_real = real_iter.next()
                data_fake = fake_iter.next()
            except StopIteration:
                break
            # -------------------------------------------------

            if data_real.shape[0] != data_fake.shape[0]:
                continue

            bz = data_real.shape[0]
            i += 1
            data = torch.cat([data_real, data_fake], dim=0)
            label = torch.cat([torch.zeros(bz, dtype=torch.long).unsqueeze(dim=0),
                               torch.ones(bz, dtype=torch.long).unsqueeze(dim=0)], dim=1).squeeze(dim=0)

            # manually shuffle
            idx = list(range(data.shape[0]))
            random.shuffle(idx)
            data = data[idx]
            label = label[idx]

            data = data.detach()
            label = label.detach()

            loss = trainer.optimize_weight(data, label)

            if trainer.total_steps % loss_freq == 0:
                logger.debug(f'loss: {loss} at step: {trainer.total_steps}')


        trainer.model.eval()
        auc, r_acc, f_acc = evaluate(trainer, dataset_img_test)
        logger.debug(
            f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
        trainer.model.train()
        trainer.epoch_step(alpha_decay)
        print("Time Cost", time.time() - s_time)
        if highest_auc < auc:
            highest_auc = auc
            trainer.save(os.path.join(ckpt_path, "saved_ck",f"{epoch}_auc_{auc}_racc_{r_acc}_facc_{f_acc}.pth"))

    trainer.model.eval()
    auc, r_acc, f_acc = evaluate(trainer, dataset_img_test)
    logger.debug(
        f'(Test @ epoch {trainer.epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')

import logging
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .xception import xception
from .efficientnet import EfficientNet
from .atp import AttentionPooling
from .model_utils import get_attention_module, get_texture_enhancer, get_auxiliary_loss
from .dct_torch1 import Laplacian
from .dct_torch2 import UnsharpMask
from .dct_torch3 import Sobel, SpatialGradient, SpatialGradient3d

class MAT_v2(nn.Module):

    def __init__(self, back_bone, num_class, pretrained, input_size:Tuple,
                attention_layers:List, atte_module_chocie, M,
                feature_layer, feature_choice, DCT_choice,
                 loss_choice, margins, alpha,
                final_dim, cb_mode,
                 rg_dropout_rate, fn_dropout_rate
                 ):
        super(MAT_v2, self).__init__()
        self.cb_mode = cb_mode
        self.prepare_cb(cb_mode, M, len(attention_layers))
        self.net = self._load_backbone(back_bone, num_class, pretrained)
        self.trial_forward_prop(input_size)
        self.attention_modules = \
            self._load_attention_modules(attention_layers, atte_module_chocie, M)
        self.atp = \
            AttentionPooling()
        self.texture_enhancer = \
            self._load_texture_enhancer(feature_layer, feature_choice, M)
        self.auxiliary_loss = \
            get_auxiliary_loss(loss_choice, self.M, self.num_features_d, num_class, alpha, margins)
        self.ensemble_fc = \
            self._load_ensemble_classifier(final_dim, num_class)
        self.projectors = \
            self._load_projectors(final_dim)
        self.regular_dp, self.final_dp = \
            self._load_dropout(rg_dropout_rate, fn_dropout_rate)
        self.DCT = self._load_dct_device(DCT_choice)

    def prepare_cb(self, mode, M, module_num):
        assert mode in ("concat", "fuse")
        self.M = M if mode == "fuse" else module_num * M
        # following the design of atp, shared by the final and the texture

    def _load_backbone(self, net_choice, num_class, pretrained):
        if 'xception' in net_choice:
            net = xception(num_class, pretrained=pretrained)
        else:   # efficientnet is used by default
            net = EfficientNet.from_pretrained(net_choice,advprop=True, num_classes=num_classes)

        # if pretrained:
        #     net.load_pretrained()

        return net

    def _load_dct_device(self, choice):
        if choice == 0:
            return None
        elif choice == 1:
            return Laplacian(kernel_size=3, border_type='reflect', normalized=True)
        elif choice == 2:
            return UnsharpMask(kernel_size=(3, 3), sigma=(1.5, 1.5), border_type='reflect')
        elif choice == 3:
            return Sobel(normalized=True, eps=1e-6)

    def _load_texture_enhancer(self, feature_layer, module_choice, M):
        assert module_choice in ['v1', 'v2']

        shape = self.get_layer_shape(feature_layer)
        texture_enhancer = \
            get_texture_enhancer(module_choice, shape, M)

        # store some params
        self.num_features = texture_enhancer.output_features
        self.num_features_d = texture_enhancer.output_features_d
        self.feature_layer = feature_layer

        return texture_enhancer

    def _load_attention_modules(self, target_layers, module_choice, M):
        assert module_choice in ['atte', 'normal', "residual"]
        module_list = []
        for tl in target_layers:
            shape = self.get_layer_shape(tl)
            module_list.append(
                get_attention_module(module_choice, shape, M)
            )
        self.atte_layers = target_layers
        return nn.Sequential(*module_list)


    def _load_ensemble_classifier(self, final_dim, num_classes):
        # final_dim is the dim of the projected features
        # * 2 is because one is the final logits and the other is the texture-enhanced feature matrix
        # both of which is projected to the dim of param['final_dim'] and the concated
        fc = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim),
            nn.Hardswish(),  # one type of activation function
            nn.Linear(final_dim, num_classes)
        )
        return fc

    def _load_projectors(self, target_dim):
        # used for texture_enhanced features
        projectors = []
        projectors.append(
            # local projector
            nn.Sequential(
                nn.Linear(self.M * self.num_features, target_dim),
                nn.Hardswish(),
                nn.Linear(target_dim, target_dim),
                nn.Hardswish(),
            )
        )
        projectors.append(
                # final projector
            nn.Sequential(
                nn.Linear(self.get_layer_shape('final')[1], target_dim),
                nn.Hardswish(),
            )
        )
        return nn.Sequential(*projectors)

    def _load_dropout(self, regular_dp_rt, final_dp_rt):
        return nn.Dropout2d(regular_dp_rt, inplace=True), \
               nn.Dropout(final_dp_rt,inplace=True)

    def train_batch(self, x, y, jump_aux=False, drop_final=False):
        fp_features = self.net(x)
        if self.feature_layer == 'logits':
            logits = fp_features['logits']
            loss = F.cross_entropy(logits,y)
            return dict(loss=loss, logits=logits)
        feature_maps = fp_features[self.feature_layer]

        if self.DCT is not None:
            feature_maps = self.DCT(feature_maps)

        attention_maps_list_ = self.cpt_atte_maps(fp_features)
        dropout_mask = self.regular_dp(
            torch.ones([attention_maps_list_[0].shape[0], self.M, 1], device=x.device))
        feature_matrix_, feature_maps_d = self.cpt_ft_mat_train(feature_maps, attention_maps_list_)
        feature_matrix = feature_matrix_ * dropout_mask

        B, M, N = feature_matrix.size()
        feature_matrix=feature_matrix.view(B, -1)
        feature_matrix=self.projectors[0](feature_matrix)

        attention_maps_list = self.dp_tensor_list(dropout_mask, attention_maps_list_)

        # get the final logits
        fn_logits = fp_features['final']
        fn_logits = self.mdf_fn_logits(fn_logits, attention_maps_list)
        fn_logits = self.final_dp(fn_logits)
        fn_logits = self.projectors[1](fn_logits)
        if drop_final:
            fn_logits *= 0
        ct_feature_matrix = torch.cat((feature_matrix, fn_logits), 1)
        ensemble_logits = self.ensemble_fc(ct_feature_matrix)
        ensemble_loss = F.binary_cross_entropy_with_logits(ensemble_logits.squeeze(1), y.float())

        # compute auxiliary loss
        self.normalize_feature(feature_maps_d)
        if not jump_aux:
            aux_loss, feature_matrix_d = self.sum_auxiliary_loss(feature_maps_d, attention_maps_list_, y)
        else:
            feature_matrix_d = self.sum_atp(feature_maps_d, attention_maps_list_)
            aux_loss = 0
        return dict(ensemble_loss=ensemble_loss,
                    aux_loss=aux_loss,
                    attention_maps=attention_maps_list_,
                    ensemble_logit=ensemble_logits,
                    feature_matrix_d=feature_matrix_d)



    def process_train(self, x, y, AG):
        if AG is None:
            return self.train_batch(x, y)
        else:
            loss_pack = self.train_batch(x, y)
            AGDA_ensemble_loss = 0
            AGDA_aux_loss = 0
            AGDA_match_loss = 0
            for attention_map in loss_pack['attention_maps']:
                with torch.no_grad():
                    Xaug, index = AG.agda(x, attention_map)
                # self.eval()
                loss_pack2 = self.train_batch(Xaug, y, jump_aux=False)
                # self.train()
                AGDA_ensemble_loss += loss_pack2['ensemble_loss']
                AGDA_aux_loss += loss_pack2['aux_loss']
                one_hot = F.one_hot(index, self.M)
                AGDA_match_loss += torch.mean(torch.norm(
                    loss_pack2['feature_matrix_d'] - loss_pack['feature_matrix_d'], dim=-1) * (torch.ones_like(one_hot) - one_hot))

            loss_pack['AGDA_ensemble_loss'] = AGDA_ensemble_loss / len(loss_pack['attention_maps'])
            loss_pack['AGDA_aux_loss'] = AGDA_aux_loss / len(loss_pack['attention_maps'])
            loss_pack['match_loss'] = AGDA_match_loss / len(loss_pack['attention_maps'])


            return loss_pack


    def forward(self, x, y=0, AG=None):
        if self.training:
            return self.process_train(x, y, AG)

        # do prediction
        # forward in the backbone
        fp_features = self.net(x)
        if self.feature_layer=='logits':
            logits=fp_features['logits']
            return logits
        # access attention maps
        attention_maps_list = self.cpt_atte_maps(fp_features)

        # enhance the texture info with attention info
        feature_maps = fp_features[self.feature_layer]
        if self.DCT is not None:
            feature_maps = self.DCT(feature_maps)
        feature_matrix = self.cpt_ft_mat(feature_maps, attention_maps_list)
        B,M,N = feature_matrix.size()
        feature_matrix = self.regular_dp(feature_matrix)
        feature_matrix=feature_matrix.view(B, -1)
        feature_matrix=self.projectors[0](feature_matrix)

        # get the final logits
        fn_logits = fp_features['final']
        fn_logits = self.mdf_fn_logits(fn_logits, attention_maps_list)
        fn_logits = self.projectors[1](fn_logits)

        # combine two
        ct_feature_matrix = torch.cat((feature_matrix, fn_logits), 1)
        ensemble_logits = self.ensemble_fc(ct_feature_matrix)
        return ensemble_logits

    def cpt_atte_maps(self, fp_features):
        attention_maps = []
        for layer_name, module in zip(self.atte_layers, self.attention_modules):
            raw_attention = fp_features[layer_name]
            attention_maps.append(
                module(raw_attention)
            )
        return attention_maps

    def cpt_ft_mat(self, feature_maps, attention_maps_list):
        feature_matrixs = []
        for attention_maps in attention_maps_list:
            feature_maps_x, _ = self.texture_enhancer(feature_maps, attention_maps)
            feature_matrix = self.atp(feature_maps_x, attention_maps)
            feature_matrixs.append(feature_matrix)

        return self.combine_features(feature_matrixs)

    def cpt_ft_mat_train(self, feature_maps, attention_maps_list):
        feature_matrixs = []
        fm_ds = []
        for attention_maps in attention_maps_list:
            feature_maps_x, fm_d = self.texture_enhancer(feature_maps, attention_maps)
            feature_matrix = self.atp(feature_maps_x, attention_maps)
            feature_matrixs.append(feature_matrix)
            fm_ds.append(fm_d)
        return self.combine_features(feature_matrixs), self.combine_features(fm_ds)

    def sum_auxiliary_loss(self, feature_maps_d, attention_maps_list, y):
        sum = 0
        fm_list = []
        for attention_maps_ in attention_maps_list:
            loss, fm_d = self.auxiliary_loss(feature_maps_d, attention_maps_, y)
            sum += loss
            fm_list.append(fm_d)

        return sum, self.combine_features(fm_list)

    def normalize_feature(self, fm):
        fm_x = fm - fm.mean(dim=[2,3],keepdim=True)
        fm_x = fm_x / (torch.std(fm_x, dim=[2,3], keepdim=True) + 1e-8)
        return fm_x

    def mdf_fn_logits(self, final_logits, attention_maps_list):
        final_logitss = []
        for attention_maps in attention_maps_list:
            attention_maps2 = attention_maps.sum(dim=1, keepdim=True)
            final = self.atp(final_logits, attention_maps2, norm=1).squeeze(1)
            final_logitss.append(final)
        return self.combine_features(final_logitss)

    def combine_features(self, ft_list):
        if self.cb_mode == "concat":
            return torch.cat(ft_list, 0)
        elif self.cb_mode == "fuse":
            return torch.sum(torch.stack(ft_list, 0), dim=0)
        #todo the dim ?

    def dp_tensor_list(self, dp_mask, tensor_list):
        new_tensor_list = []
        for tensor in tensor_list:
            masked_tensor = torch.unsqueeze(dp_mask, -1) * tensor
            new_tensor_list.append(masked_tensor)
        return new_tensor_list

    def trial_forward_prop(self, input_size):
        with torch.no_grad():
            self.trial_result:Dict = self.net(torch.zeros(1,3,input_size[0], input_size[1]))

    def get_layer_shape(self, name):
        return self.trial_result.get(name, None).shape

    def freeze(self, freeze_list):
        for i in freeze_list:
            if 'backbone' in i:
                self.net.requires_grad_(False)
            elif 'attention' in i:
                self.attention_modules.requires_grad_(False)
            elif 'feature_center' in i:
                self.auxiliary_loss.alpha = 0
            elif 'texture_enhance' in i:
                self.texture_enhancer.requires_grad_(False)
            elif 'fcs' in i:
                self.projectors.requires_grad_(False)
                self.ensemble_fc.requires_grad_(False)
            else:
                if 'xception' in str(type(self.net)):
                    for j in self.net.seq:
                        if j[0] == i:
                            for t in j[1]:
                                t.requires_grad_(False)

                if 'EfficientNet' in str(type(self.net)):
                    if i == 'b0':
                        self.net._conv_stem.requires_grad_(False)
                    stage_map = self.net.stage_map
                    for c in range(len(stage_map) - 2, -1, -1):
                        if not stage_map[c]:
                            stage_map[c] = stage_map[c + 1]
                    for c1, c2 in zip(stage_map, self.net._blocks):
                        if c1 == i:
                            c2.requires_grad_(False)


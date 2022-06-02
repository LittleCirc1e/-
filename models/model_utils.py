
from .attention_modules import *
from .texture_enhancer import *
from .auxiliary_loss import *
from .atp import *

def get_attention_module(net_choice, shape, M):
    net_map = {
        'normal':NormalAttMod,
        'atte':RanAttMod,
        'residual':ResidualAttMod
    }
    return net_map[net_choice](shape[1], M)

def get_texture_enhancer(net_choice, shape, M):
    net_map = {
        'v1':Texture_Enhance_v1,
        'v2':Texture_Enhance_v2,
    }

    return net_map[net_choice](shape[1], M)


def get_auxiliary_loss(loss_choice, M, num_feature, num_class, alpha, margins):
    """
    :param margins:
        margins[1] is the "inner margin"
    """
    loss_map = {
        'v1':Auxiliary_Loss_v1,
        'v2':Auxiliary_Loss_v2
    }

    return loss_map[loss_choice](M, num_feature, num_class, alpha, margins[0], margins[1])

from qkformer_tawq import qkformer_tawq
from spikingformer_tawq import spikingformer_tawq
import torch.nn as nn
from functools import partial



def spikingformer_tawq_8_384(T=4, **kwargs):
    model = spikingformer_tawq(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=384, num_heads=6, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        **kwargs
    )
    return model

def spikingformer_tawq_8_512(T=4, **kwargs):
    model = spikingformer_tawq(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=512, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        **kwargs
    )
    return model

def spikingformer_tawq_8_768(T=4, **kwargs):
    model = spikingformer_tawq(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        **kwargs
    )
    return model


def qkformer_tawq_10_768(T=4, **kwargs):
    model = qkformer_tawq(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=10, sr_ratios=1,
        **kwargs
    )
    return model

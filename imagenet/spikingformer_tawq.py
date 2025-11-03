import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from tawq import QConv2d

__all__ = ['spikingformer_tawq']


class AConv3x3_stem(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.conv3x3 = nn.Conv2d(dim_in, dim_out, 3, 1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)
        if self.stride ==2:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv3x3(x.flatten(0, 1))).reshape(T, B, self.dim_out, H, W)
        if self.stride == 2:
            x = self.pool(x.flatten(0, 1)).reshape(T, B, self.dim_out, H//2, W//2)
        return x


class Conv3x3(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.conv3x3 = QConv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)
        if self.stride ==2:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif(x)
        x = self.conv3x3(x)
        if self.stride == 2:
            x = self.pool(self.bn(x.flatten(0, 1))).reshape(T, B, self.dim_out, H//2, W//2)
        else:
            x = self.bn(x.flatten(0, 1)).reshape(T, B, self.dim_out, H, W)
        return x


class DWConv3x3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.conv3x3 = QConv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif(x)
        x = self.conv3x3(x)
        x = self.bn(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x
    

class Conv1x1(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.conv1x1 = QConv2d(dim_in, dim_out, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif(x)
        x = self.conv1x1(x)
        x = self.bn(x.flatten(0, 1)).reshape(T, B, self.dim_out, H//self.stride, W//self.stride)
        return x
    

class Conv1x1_ssa(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.conv1x1 = QConv2d(dim_in, dim_out, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.conv1x1(x)
        x = self.bn(x.flatten(0, 1)).reshape(T, B, self.dim_out, H//self.stride, W//self.stride)
        x = self.lif(x)
        return x


class Conv1x1_proj(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.conv1x1 = QConv2d(dim_in, dim_out, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.conv1x1(x)
        x = self.bn(x.flatten(0, 1)).reshape(T, B, self.dim_out, H//self.stride, W//self.stride)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.mlp1 = Conv1x1(in_features, hidden_features)
        self.mlp2 = Conv1x1(hidden_features, out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.pre_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.q_conv = Conv1x1_ssa(dim, dim)
        self.k_conv = Conv1x1_ssa(dim, dim)
        self.v_conv = Conv1x1_ssa(dim, dim)

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')
        self.proj_conv = Conv1x1_proj(dim, dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.pre_lif(x)

        q_conv_out = self.q_conv(x).flatten(3)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x).flatten(3)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x).flatten(3)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125
        x = x.transpose(3, 4).reshape(T, B, C, -1).reshape(T, B, C, H, W)

        x = self.attn_lif(x)
        x = self.proj_conv(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = AConv3x3_stem(in_channels, out_channels//8, stride=2)
    def forward(self, x):
        x = self.conv(x)
        return x


class Tokenizer(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = Conv3x3(out_channels//8, out_channels//4, stride=2)
        self.conv2 = Conv3x3(out_channels//4, out_channels//2, stride=2)
        self.conv3 = Conv3x3(out_channels//2, out_channels, stride=2)
        self.conv4 = Conv3x3(out_channels, out_channels, stride=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x



class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ssa = SSA(dim, num_heads=num_heads)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.ssa(x)
        x = x + self.mlp(x)
        return x
    

class spikingformer_tawq(nn.Module):
    def __init__(self,
                 T=4,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2],
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.num_patches = 64

        self.stem = Stem(in_channels, embed_dims)
        self.tokenizer = Tokenizer(embed_dims)
        self.stage3 = nn.Sequential(*[SpikingTransformer(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios)
            for i in range(depths)]) # self.blocks

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[1]
        x = self.stem(x)
        x = self.tokenizer(x)
        x = self.stage3(x).flatten(3).mean(3)
        return x

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


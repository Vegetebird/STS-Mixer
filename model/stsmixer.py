import os
import sys
import math
import torch
import numpy as np
from torch import nn
import pytorch3d.ops
import pytorch3d.utils
from einops import rearrange
import torch.nn.functional as F
from model.module.point_4d_convolution import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


@torch.no_grad()
def eig_vector(data, K=10):
    b, n, _ = data.shape
    _, idx, _ = pytorch3d.ops.knn_points(data, data, K=K)

    idx0 = torch.arange(0,b,device=data.device).reshape((b,1)).expand(-1,n*K).reshape((1,b*n*K))
    idx1 = torch.arange(0,n,device=data.device).reshape((1,n,1)).expand(b,n,K).reshape((1,b*n*K))
    idx = idx.reshape((1,b*n*K))
    idx = torch.cat([idx0, idx1, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=bool, device=data.device)
    A = torch.sparse_coo_tensor(idx, ones, (b, n, n)).to_dense()
    A = A | A.transpose(1, 2)
    A = A.float()
    deg = torch.diag_embed(torch.sum(A, dim=2))
    laplacian = deg - A

    u, v = torch.linalg.eig(laplacian)
    
    return v.real, laplacian, u.real


def gft(x, K):
    b, n, c = x.shape
    v, laplacian, u = eig_vector(x, K)

    x = torch.einsum('bij,bjk->bik',v.transpose(1,2), x) 

    return x, v


def igft(x, v, factor=None):
    x = torch.einsum('bij,bjk->bik',v, x)

    return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads

        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, features):
        b, f, n, c, h = *features.shape, self.heads

        norm_features = self.norm(features)
        qkv = self.to_qkv(norm_features).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b f n (h d) -> b h (f n) d', h = h), qkv)

        x_flatten = rearrange(x, 'b f n d -> b (f n) d')

        delta_x = torch.unsqueeze(input=x_flatten, dim=1) - torch.unsqueeze(input=x_flatten, dim=2)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        v = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        attn = torch.unsqueeze(input=attn, dim=4)
        delta_x = torch.unsqueeze(input=delta_x, dim=1)
        delta_x = torch.sum(input=attn*delta_x, dim=3, keepdim=False)

        displacement_features = self.spatial_op(delta_x)

        out = v + displacement_features
        out = rearrange(out, 'b h m d -> b m (h d)')
        out =  self.to_out(out)
        out = rearrange(out, 'b (f n) d -> b f n d', f=f, n=n)
        
        return out + features

class Transformer_fre(nn.Module):
    def __init__(self, args, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dropout = dropout),
                Attention(dim, heads = heads, dropout = dropout),
                Attention(dim, heads = heads, dropout = dropout),

                FeedForward(dim * 3, mlp_dim * 3, dropout = dropout),
            ]))

    def forward(self, xyzs_low, xyzs_mid, xyzs_high, x):
        x = x.permute(0, 1, 3, 2)

        c = x.shape[-1]

        x_low  = x
        x_mid  = x
        x_high = x

        for attn_low, attn_mid, attn_high, ff in self.layers:
            x_low  = attn_low(xyzs_low, x_low)
            x_mid  = attn_mid(xyzs_mid, x_mid)
            x_high = attn_high(xyzs_high, x_high)

            x = torch.cat([x_low, x_mid, x_high], dim = -1)

            x = ff(x)

            x_low, x_mid, x_high = x[:, :, :, :c], x[:, :, :, c:2*c], x[:, :, :, 2*c:]

        return x_low, x_mid, x_high

class Transformer(nn.Module):
    def __init__(self, args, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
            
    def forward(self, xyzs, features):
        features = features.permute(0, 1, 3, 2)
        
        for attn, ff in self.layers:
            features = attn(xyzs, features)
            features = ff(features)
        return features


class Model(nn.Module):
    def __init__(self, args, radius=0.3, nsamples=32, spatial_stride=32, temporal_kernel_size=3, temporal_stride=2,
                 dim=128, depth=3, heads=8, mlp_dim=2048, num_classes=20, dropout1=0.0, dropout2=0.0):
        super().__init__()

        mlp_dim = dim * 2
        self.K = 10
        self.low_num = 6
        self.high_num = 10

        self.embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                            spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                            temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, 
                            temporal_padding=[1, 0], operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer_fre(args, dim, depth, heads, mlp_dim, dropout=dropout1)

        self.head = nn.Sequential(
            nn.LayerNorm(dim*3),
            nn.Linear(dim*3, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x):
        b, f, n, c = x.shape

        x, features = self.embedding(x)

        b, f, n, c = x.shape

        x_gft = rearrange(x, 'b f n c -> (b f) n c')
        x_gft, v = gft(x_gft, self.K)

        x_low = x_gft.clone()
        x_low[:, self.low_num:] = 0
        x_low = igft(x_low, v)

        x_mid = x_gft.clone()
        x_mid[:, :self.low_num] = 0
        x_mid[:, self.high_num:] = 0
        x_mid = igft(x_mid, v)

        x_high = x_gft.clone()
        x_high[:, :self.high_num] = 0
        x_high = igft(x_high, v)

        x_low  = rearrange(x_low, '(b f) n c -> b f n c', f=f)
        x_mid  = rearrange(x_mid, '(b f) n c -> b f n c', f=f)
        x_high = rearrange(x_high, '(b f) n c -> b f n c', f=f)

        x_low, x_mid, x_high = self.transformer(x_low, x_mid, x_high, features)

        x_low = torch.max(input=x_low, dim=1, keepdim=False, out=None)[0]
        x_low = torch.max(input=x_low, dim=1, keepdim=False, out=None)[0]

        x_mid = torch.max(input=x_mid, dim=1, keepdim=False, out=None)[0]
        x_mid = torch.max(input=x_mid, dim=1, keepdim=False, out=None)[0]

        x_high = torch.max(input=x_high, dim=1, keepdim=False, out=None)[0]
        x_high = torch.max(input=x_high, dim=1, keepdim=False, out=None)[0]

        x = torch.cat([x_low, x_mid, x_high], dim=-1)

        x = self.head(x)

        return x

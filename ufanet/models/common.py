import torch
import numpy as np
import torch.nn as nn
from torchvision.transforms import Resize


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.cnn_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cnn_down(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))
        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class SelfAttentionResized(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.height = int(np.sqrt(dim))
        self.heads = heads
        self.self_attention = SelfAttention(self.dim, self.heads)
        self.res_down = Resize((self.height, self.height))
        self.res_up = None

    def forward(self, batched_data):
        b, c, h, w = batched_data.shape
        self.res_up = Resize((h - 2, w - 2))
        batched_data = self.res_down(batched_data)
        batched_data = batched_data.view(b, c, -1)
        batched_data = self.self_attention(batched_data)
        batched_data = batched_data.view(b, c, self.height, self.height)
        return self.res_up(batched_data)


class ATBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dim=1024, heads=8):
        super().__init__()
        self.attn_block = SelfAttentionResized(dim, heads)
        self.conv_block = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, batched_data):
        return self.relu(self.conv_block(self.relu(self.attn_block(batched_data))))


class FeatureAggregation(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.avg_pool = nn.AvgPool2d(2)
        self.high_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.low_conv = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.fwd_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)  # padding set to 1 stabilize the image shape

    def forward(self, high_level_features, low_level_features):
        _, _, h, w = low_level_features.shape
        res = Resize((h, w))
        return self.avg_pool(self.fwd_conv(res(self.high_conv(self.avg_pool(high_level_features)))
                                           + self.low_conv(low_level_features)))

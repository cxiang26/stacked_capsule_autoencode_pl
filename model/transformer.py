import torch
from torch import nn

import math
from .util import normal_init

class SetTransformer(nn.Module):
    def __init__(self,
                 n_layers=3,
                 n_heads=1,
                 n_dims=16,
                 n_output_dims=256,
                 n_outputs=10,
                 layer_norm=True,
                 dropout_rate=0.,):
        super(SetTransformer, self).__init__()
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._n_dims = n_dims
        self._n_output_dims = n_output_dims
        self._n_outputs = n_outputs
        self._ln = layer_norm
        self._dropout_rate = dropout_rate
        self.input_layer = nn.Linear(144, self._n_dims)
        self.sab = nn.Sequential()
        for i in range(self._n_layers):
            self.sab.add_module('sab_{}'.format(i), SAB(n_heads, self._n_dims, self._n_dims, self._n_dims, self._dropout_rate, ln=True))
        self.output_layer = nn.Linear(self._n_dims, self._n_output_dims)
        self.isab = ISAB(self._n_output_dims, self._n_outputs, self._n_output_dims)
        self._init_weights()

    def _init_weights(self):
        normal_init(self.input_layer, std=0.001)
        normal_init(self.output_layer, std=0.001)

    def forward(self, x, presence=None):
        h = self.input_layer(x)
        for i in range(self._n_layers):
            h = self.sab[i](h, presence)
        z = self.output_layer(h)
        return self.isab(z, presence)

class SAB(nn.Module):
    def __init__(self, n_heads, dim_Q=16, dim_K=16, dim_V=16, dropout_rate=0., ln=False):
        super(SAB, self).__init__()
        self._n_heads = n_heads
        self._dropout_rate = dropout_rate
        self._dim_V = dim_V
        self._mab = MAB(dim_Q, dim_K, dim_V, n_heads)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.mlp1 = nn.Linear(dim_V, 2*dim_V)
        self.mlp2 = nn.Linear(2*dim_V, dim_V)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        normal_init(self.fc_o, std=0.001)
        normal_init(self.mlp1, std=0.001)
        normal_init(self.mlp2, std=0.001)

    def forward(self, x, presence=None):
        y = self._mab(x, x, x, presence)
        if self._dropout_rate > 0.:
            x = torch.dropout(x, p=self._dropout_rate)
        y += x
        if presence is not None:
            y *= presence
        y = y if getattr(self, 'ln0', None) is None else self.ln0(y)
        h = self.mlp2(self.relu(self.mlp1(y)))
        if self._dropout_rate > 0.:
            h = torch.dropout(h, p=self._dropout_rate)
        h += y
        h = h if getattr(self, 'ln1', None) is None else self.ln1(h)
        return h

class ISAB(nn.Module):
    def __init__(self, dim_in=16, num_inds=10, dim_out=256):
        super(ISAB, self).__init__()
        self.inducing_points = nn.Parameter(torch.rand(1, num_inds, dim_out))
        self.fc_q = nn.Linear(dim_in, dim_out)
        self.fc_k = nn.Linear(dim_in, dim_out)
        self.fc_v = nn.Linear(dim_in, dim_out)
        self.qkv_att = QKVAttention()
        self.fc_o = nn.Linear(dim_out, dim_out)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.inducing_points.data, std=0.001)
        normal_init(self.fc_q, std=0.001)
        normal_init(self.fc_k, std=0.001)
        normal_init(self.fc_v, std=0.001)
        normal_init(self.fc_o, std=0.001)

    def forward(self, x, presence=None):
        inducing_points = self.inducing_points.repeat(x.size(0), 1, 1)
        q = self.fc_q(inducing_points)
        k = self.fc_k(x)
        v = self.fc_v(x)
        res = self.qkv_att(q, k, v, presence)
        return self.fc_o(res)

class MAB(nn.Module):
    def __init__(self, dim_q=16, dim_k=16, dim_v=16, n_heads=1):
        super(MAB, self).__init__()
        self._dim_v = dim_v
        self._n_heads = n_heads
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_v, dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)

    def forward(self, q, k, v, presence=None):
        q = self.fc_q(q)
        k, v = self.fc_k(k), self.fc_v(v)
        dim_split = self._dim_v // self._n_heads
        q_ = torch.cat(q.split(dim_split, 2), 0)
        k_ = torch.cat(k.split(dim_split, 2), 0)
        v_ = torch.cat(v.split(dim_split, 2), 0)

        routing = q_.bmm(k_.transpose(1,2))
        if presence is not None:
            routing -= (1. - presence.transpose(1, 2)) * 1e32
        A = torch.softmax(routing/math.sqrt(self._dim_v), -1)
        res = A.bmm(v_)
        res = self.fc_o(res)
        return res

class QKVAttention(nn.Module):
    def __init__(self):
        super(QKVAttention, self).__init__()
    def forward(self, q, k, v, presence=None):
        """
        :param q: Tensor of shape [B, N, d_k]
        :param k: Tensor of shape [B, M, d_k]
        :param v: Tensor of shape [B, M, d_v]
        :param presence: None or tensor of shape [B, M]
        :return: Tensor of shape [B, N, d_v]
        """
        _, _, n_dim = q.size()
        routing = torch.matmul(q, k.transpose(1,2))
        routing = routing/math.sqrt(n_dim)
        if presence is not None:
            routing -= (1. - presence.transpose(1, 2)) * 1e32
        routing = torch.softmax(routing, dim=-1)
        res = torch.matmul(routing, v)
        return res
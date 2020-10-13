import torch
from torch import nn
import torch.nn.functional as F

from .util import normal_init, geometric_transform
from monty.collections import AttrDict

class CapsuleLayer(nn.Module):
    _n_transform_params = 6
    def __init__(self, n_caps, n_caps_dims, n_votes, n_caps_params=32,
                 n_hiddens=128, learn_vote_scale=True, deformations=True,
                 noise_scale=4., similarity_transform=True,
                 caps_dropout_rate=0.0):
        super(CapsuleLayer, self).__init__()
        self._n_caps = n_caps
        self._n_caps_dims = n_caps_dims
        self._n_caps_params = n_caps_params
        self._n_votes = n_votes
        self._n_hiddens = n_hiddens
        self._learn_vote_scale = learn_vote_scale
        self._deformations = deformations
        self._noise_scale = noise_scale

        self._n_outputs = (self._n_votes * self._n_transform_params + self._n_transform_params
                           + 1 + self._n_votes + self._n_votes)

        self._similarity_transform = similarity_transform
        self._caps_dropout_rate = caps_dropout_rate

        # self._n_caps MLPs, one for every object capsule, which predicts capsule parameters from Set Transformer’s outputs
        self.batch_mlp_w1 = nn.Parameter(0.5*(torch.rand(size=(1, self._n_caps, 256, self._n_hiddens))-0.5))
        self.batch_mlp_b1 = nn.Parameter(torch.zeros(size=(1, self._n_caps, 1, self._n_hiddens)))
        self.batch_mlp_w2 = nn.Parameter(0.5*(torch.rand(size=(1, self._n_caps, self._n_hiddens, self._n_caps_params))-0.5))
        self.batch_mlp_b2 = nn.Parameter(torch.zeros(size=(1, self._n_caps, 1, self._n_caps_params)))

        self.batch_caps_mlp_w = nn.Parameter(0.5*(torch.rand(size=(1, self._n_caps, self._n_caps_params+1, self._n_hiddens))-0.5))
        self.batch_caps_mlp_b = nn.Parameter(torch.zeros(size=(1, self._n_caps, 1, self._n_hiddens)))

        # cpr_dynamic no bias
        self.batch_cpr_dynamic_w = nn.Parameter(0.5*(torch.rand(size=(1, self._n_caps, self._n_hiddens, self._n_votes * self._n_transform_params))-0.5))

        self.batch_ccr_w = nn.Parameter(0.5*(torch.rand(size=(1, self._n_caps, self._n_hiddens, self._n_transform_params))-0.5))
        self.batch_ccr_b = nn.Parameter(torch.zeros(size=(1, self._n_caps, 1, self._n_transform_params)))

        self.batch_pres_logit_per_caps_w = nn.Parameter(0.5*(torch.rand(size=(1, self._n_caps, self._n_hiddens, 1))-0.5))
        self.batch_pres_logit_per_caps_b = nn.Parameter(torch.zeros(size=(1, self._n_caps, 1, 1)))

        self.batch_pres_logit_per_vote_w = nn.Parameter(0.5*(torch.rand(size=(1, self._n_caps, self._n_hiddens, self._n_votes))-0.5))
        self.batch_pres_logit_per_vote_b = nn.Parameter(torch.zeros(size=(1, self._n_caps, 1, self._n_votes)))

        self.batch_scale_per_vote_w = nn.Parameter(0.5*(torch.rand(size=(1, self._n_caps, self._n_hiddens, self._n_votes))-0.5))
        self.batch_scale_per_vote_b = nn.Parameter(torch.zeros(size=(1, self._n_caps, 1, self._n_votes)))

        self.cpr_static = nn.Parameter(torch.rand(1, self._n_caps, self._n_votes, self._n_transform_params)-0.5)

        assert n_caps_dims == 2, ('This is the only value implemented now due to the restriction of similarity transform.')

    def _init_weights(self):
        for m in self.caps_mlp:
            if type(m) == nn.Linear:
                normal_init(m, std=0.05)
        normal_init(self.mlp1, std=0.05)
        normal_init(self.mlp2, std=0.05)
        normal_init(self.cpr_dynamic, std=0.05)
        normal_init(self.ccr, std=0.05)
        normal_init(self.pres_logit_per_vote, std=0.05)
        normal_init(self.pres_logit_per_caps, std=0.05)
        normal_init(self.scale_per_vote, std=0.05)
        normal_init(self.cpr_dynamic, std=0.05)

    def forward(self, features, parent_presence=None):
        batch_size = features.size(0)
        batch_shape = [batch_size, self._n_caps]

        if self._n_caps_params is not None:
            raw_caps_params = features.unsqueeze(dim=2)
            raw_caps_params = raw_caps_params @ self.batch_mlp_w1 + self.batch_mlp_b1
            raw_caps_params = torch.relu(raw_caps_params)
            raw_caps_params = torch.relu(raw_caps_params @ self.batch_mlp_w2 + self.batch_mlp_b2)
            caps_params = raw_caps_params
        else:
            caps_params = features

        if self._caps_dropout_rate == 0.:
            caps_exist = torch.ones(batch_shape + [1] + [1]).to(caps_params.device)
        else:
            caps_exist = (torch.rand(batch_shape + [1] + [1]) > self._caps_dropout_rate).to(caps_params.device)

        caps_params = torch.cat([caps_params, caps_exist], -1)

        output_shapes = (
            [self._n_votes, self._n_transform_params],  # CPR_dynamic
            [1, self._n_transform_params],  # CCR
            [1],  # per-capsule presence
            [self._n_votes],  # per-vote-presence
            [self._n_votes],  # per-vote scale
        )

        # caps_feat = self.caps_mlp(caps_params)
        caps_feat = caps_params @ self.batch_caps_mlp_w + self.batch_caps_mlp_b
        caps_feat = torch.relu(caps_feat)

        cpr_dynamic         = caps_feat @ self.batch_cpr_dynamic_w
        ccr                 = caps_feat @ self.batch_ccr_w + self.batch_ccr_b
        pres_logit_per_caps = caps_feat @ self.batch_pres_logit_per_caps_w + self.batch_pres_logit_per_caps_b
        pres_logit_per_vote = caps_feat @ self.batch_pres_logit_per_vote_w + self.batch_pres_logit_per_vote_b
        scale_per_vote      = caps_feat @ self.batch_scale_per_vote_w + self.batch_scale_per_vote_b
        cpr_dynamic = cpr_dynamic.view(batch_shape + output_shapes[0])
        # ccr = ccr.view(batch_shape + output_shapes[1])
        pres_logit_per_caps = pres_logit_per_caps.squeeze(dim=-2)
        pres_logit_per_vote = pres_logit_per_vote.squeeze(dim=-2)
        scale_per_vote = scale_per_vote.squeeze(dim=-2)

        # if self._caps_dropout_rate != 0.0:
        #     pres_logit_per_caps += caps_exist

        if self._noise_scale > 0.:
            pres_logit_per_caps += ((torch.rand_like(pres_logit_per_caps) - .5) * self._noise_scale)
            pres_logit_per_vote += ((torch.rand_like(pres_logit_per_vote) - .5) * self._noise_scale)

        ccr = geometric_transform(ccr, as_matrix=True)

        if not self._deformations:
            cpr_dynamic = torch.zeros_like(cpr_dynamic)

        cpr = geometric_transform(cpr_dynamic + self.cpr_static, as_matrix=True)
        votes = torch.matmul(ccr, cpr)  # B, n_classes, n_votes, 3, 3

        if parent_presence is not None:
            pres_per_caps = parent_presence
        else:
            pres_per_caps = torch.sigmoid(pres_logit_per_caps)

        pres_per_vote = pres_per_caps * torch.sigmoid(pres_logit_per_vote)

        if self._learn_vote_scale:
            scale_per_vote = F.softplus(scale_per_vote + .5) + 1e-2
        else:
            scale_per_vote = torch.zeros_like(scale_per_vote) + 1.
        dynamic_weights_l2 = torch.sum(cpr_dynamic ** 2) / batch_size / 2.
        return AttrDict(
        vote=votes,
        scale=scale_per_vote,
        vote_presence=pres_per_vote,
        pres_logit_per_caps=pres_logit_per_caps,
        pres_logit_per_vote=pres_logit_per_vote,
        dynamic_weights_l2=dynamic_weights_l2,
        raw_caps_params=raw_caps_params.squeeze(dim=-2),
        raw_caps_features=features,
    )
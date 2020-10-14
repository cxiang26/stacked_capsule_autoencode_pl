import torch
from torch import nn
import torch.nn.functional as F
from .util import safe_log, geometric_transform, normal_init
from .transformer import SetTransformer
from .capsule import CapsuleLayer
from .distributions import GaussianMixture

import collections
from monty.collections import AttrDict

class SCAE(nn.Module):
    def __init__(self,
                 primary_encoder,
                 primary_decoder,
                 encoder,
                 decoder,
                 n_classes=None,
                 ):
        super(SCAE, self).__init__()
        self._primary_encoder = primary_encoder
        self._primary_decoder = primary_decoder
        self._encoder = encoder
        self._decoder = decoder
        self._n_classes = n_classes

    def forward(self, x):
        primary_caps = self._primary_encoder(x) # pose: (B, M, 6), pres:(B, M)
        pose, pres, feat = primary_caps.pose, primary_caps.presence, primary_caps.feature
        pose = geometric_transform(pose).view(*pose.size()[:2], -1)
        input_pose = torch.cat([pose, 1. - pres], -1)

        input_pres = pres.detach()
        input_pose = input_pose.detach()

        target_pose = pose.detach()
        target_pres = pres.detach()

        input_pose = torch.cat([input_pose, feat], dim=-1)
        templates = self._primary_decoder.make_templates() # (B, num_temp, 11, 11)

        ## flatten templates and concat to input_pose
        templates_flatten = templates.view(templates.size(0), -1).detach() # templates as obj_encoder should be detached here
        pose_with_templates = torch.cat([input_pose, templates_flatten.repeat(input_pose.size(0), 1, 1)], dim=-1)
        h = self._encoder(pose_with_templates, input_pres)

        res, res_ = self._decoder(h, target_pose, target_pres)
        res.update(res_._asdict())

        res.primary_presence = primary_caps.presence
        primary_dec_vote = primary_caps.pose
        primary_dec_pres = primary_caps.presence

        # res.bottom_up_rec = self._primary_decoder(primary_caps.pose,
        #                                           primary_caps.presence)

        # res.top_down_rec = self._primary_decoder(res.winner,
        #                                          primary_caps.presence)

        rec = self._primary_decoder(primary_dec_vote,
                                    primary_dec_pres)

        # n_caps = res.vote.size(1)
        # tiled_presence = primary_caps.presence.repeat(n_caps, 1, 1)
        # tiled_feature = primary_caps.feature.repeat(n_caps, 1, 1)
        # tiled_img_embedding = primary_caps.img_embedding.repeat(n_caps, 1, 1, 1)

        # res.top_down_per_caps_rec = self._primary_decoder(res.vote.view(-1, *res.vote.size()[-2:]),
        #                                                   res.vote_presence.view(-1, *res.vote_presence.size()[-1:]).unsqueeze(dim=-1) * tiled_presence)

        # res.templates = templates
        # res.template_pres = pres

        B = x.size(0)
        # expanded_x = x.unsqueeze(dim=1)
        res.rec_ll_per_pixel = rec.pdf.log_prob(x) # (x: B, 1, 1, 28, 28)
        # res.rec_ll_per_pixel = torch.logsumexp(res.rec_ll_per_pixel + rec.template_mixing_log_prob, dim=1)
        res.rec_ll = res.rec_ll_per_pixel.view(B, -1).sum(-1).mean()

        res.primary_caps_l1 = res.primary_presence.view(B, -1).abs().sum(dim=-1).mean()
        res.transformed_templates = rec.transformed_templates
        res.rec = rec
        return res

class part_encoder(nn.Module):
    OutputTuple = collections.namedtuple(  # pylint:disable=invalid-name
        'PrimaryCapsuleTuple',
        'pose feature presence presence_logit '
        'img_embedding')
    def __init__(self, num_capsules=24, hidden_layers=[(1, 128, 2), (128, 128, 2), (128, 128, 1), (128, 128, 1)], noise_scale=4.):
        super(part_encoder, self).__init__()
        self._pose = 6
        self._pred = 1
        self._feat = 16
        self._num_caps = num_capsules
        self._noise_scale = noise_scale
        self._hidden_layers = hidden_layers
        self.cnn_encoder = nn.Sequential()
        for i, (hl_in, hl_out, str) in enumerate(self._hidden_layers):
            self.cnn_encoder.add_module('cnn_encoder_cov_{}'.format(i), nn.Conv2d(hl_in, hl_out, 3, stride=str))
            self.cnn_encoder.add_module('cnn_encoder_act_{}'.format(i), nn.ReLU())
        self.part_pose = nn.Conv2d(self._hidden_layers[-1][1], self._num_caps * self._pose, 1)
        self.part_pred = nn.Conv2d(self._hidden_layers[-1][1], self._num_caps * self._pred, 1)
        self.part_feat = nn.Conv2d(self._hidden_layers[-1][1], self._num_caps * self._feat, 1)
        self.attention = nn.Conv2d(self._hidden_layers[-1][1], self._num_caps * 1, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1) # dim=-1
        self._init_weights()

    def _init_weights(self):
        for m in self.cnn_encoder:
            if type(m) == nn.Conv2d:
                normal_init(m, std=0.05)
        normal_init(self.part_pose, std=0.05)
        normal_init(self.part_pred, std=0.05)
        normal_init(self.part_feat, std=0.05)
        normal_init(self.attention, std=0.05)

    def forward(self, x):
        output = self.cnn_encoder(x)
        B = output.size(0)
        attention = self.attention(output).view(B, self._num_caps, -1)
        pose = self.part_pose(output).view(B, self._num_caps * self._pose, -1)
        pred = self.part_pred(output).view(B, self._num_caps * self._pred, -1)
        feat = self.part_feat(output).view(B, self._num_caps * self._feat, -1)

        attention = self.softmax(attention).view(B, self._num_caps, 1, -1) # (B, self._num_caps, 4) for mnist
        pose = (pose.view(B, self._num_caps, self._pose, -1) * attention).sum(dim=-1)
        pres_logit = (pred.view(B, self._num_caps, self._pred, -1) * attention).sum(dim=-1)
        feat = (feat.view(B, self._num_caps, self._feat, -1) * attention).sum(dim=-1)

        if self._noise_scale > 0.:
            pres_logit += ((torch.rand_like(pres_logit) - .5) * self._noise_scale)

        pose = self.relu(pose)
        pres = self.sigmoid(pres_logit)
        feat = self.relu(feat)
        return self.OutputTuple(pose, feat, pres, pres_logit, output)

class part_decoder(nn.Module):
    def __init__(self, num_capsules=24, template_size=11, target_size=28):
        super(part_decoder, self).__init__()
        self._num_templates = num_capsules
        self._num_capsules = num_capsules
        self._template_size = template_size
        self._target_size = target_size
        self._bg_image = nn.Parameter(torch.rand(1))
        self._temperature_logit = nn.Parameter(torch.rand(1))
        self.templates = nn.Parameter(torch.rand(self._num_templates, 1, template_size, template_size))
        self._init_weights()


    def _init_weights(self):
        nn.init.normal_(self._bg_image.data)
        nn.init.normal_(self._temperature_logit.data)
        nn.init.orthogonal_(self.templates.data.view(self._num_templates, -1))

    def make_templates(self):
        return F.relu6(self.templates * 6.) / 6. # make sure value belong to [0, 1]

    def forward(self, pose, presence=None, bg_image=None):
        B, _, _ = pose.size()
        templates = self.make_templates()
        transformed_templates = [F.grid_sample(templates[i].repeat(B, 1, 1, 1),
                                               # sce.to(device) could not transfrom self.templates to "cuda"
                                               F.affine_grid(
                                                   geometric_transform(pose[:, i, :]),  # pose
                                                   torch.Size((B, 1, self._target_size, self._target_size)),  # size
                                                  align_corners=True
                                               ),
                                               align_corners=True).unsqueeze(dim=1)
                                 for i in range(self._num_capsules)]
        # shape: (B, self._num_capsules, 1, template_size, template_size)
        transformed_templates = torch.cat(transformed_templates, 1)
        if bg_image is not None:
            bg_image = bg_image.unsqueeze(dim=1)
        else:
            bg_image = torch.sigmoid(self._bg_image)
            bg_image = torch.zeros_like(transformed_templates[:, :1] + bg_image)

        transformed_templates = torch.cat([transformed_templates, bg_image], dim=1)
        if presence is not None:
            presence = torch.cat([presence, torch.ones(presence.size(0), 1, 1).to(presence.device)], dim=1)

        if True:
            temperature = F.softplus(self._temperature_logit + 0.5) + 1e-4
            template_mixing_logits = transformed_templates / temperature
        # template_mixing_logits = template_mixing_logits.max(dim=1, keepdim=True).values  # allowing occlusion by other templates
        scale = 1. # constant variance
        presence = safe_log(presence)
        template_mixing_logits = template_mixing_logits + presence.unsqueeze(dim=-1).unsqueeze(dim=-1)
        # template_mixing_log_prob = template_mixing_logits - torch.logsumexp(template_mixing_logits, 1, keepdim=True)
        # pdf = torch.distributions.Normal(transformed_templates, scale)
        pdf = GaussianMixture.make_from_stats(
            loc=transformed_templates,
            scale=scale,
            mixing_logits=template_mixing_logits
        )
        return AttrDict(transformed_templates=transformed_templates[:, :-1],
                        template_mixing_logits=template_mixing_logits,
                        scale=scale,
                        pdf=pdf)

class obj_encoder(nn.Module):
    def __init__(self,n_layers=3,
                 n_heads=1,
                 n_dims=16,
                 n_output_dims=256,
                 n_outputs=10,
                 layer_norm=True,
                 dropout_rate=0.):
        super(obj_encoder, self).__init__()
        self.encoder = SetTransformer(n_layers,
                 n_heads,
                 n_dims,
                 n_output_dims,
                 n_outputs,
                 layer_norm,
                 dropout_rate)

    def forward(self, x, presence=None):
        x = self.encoder(x, presence)
        return x

class obj_decoder(nn.Module):
    OutputTuple = collections.namedtuple('CapsuleLikelihoodTuple',  # pylint:disable=invalid-name
                                         ('log_prob vote_presence winner '
                                          'winner_pres is_from_capsule '
                                          'mixing_logits mixing_log_prob '
                                          'soft_winner soft_winner_pres '
                                          'posterior_mixing_probs'))
    def __init__(self, n_caps, n_caps_dims, n_votes, **capsule_kwargs):
        super(obj_decoder, self).__init__()
        self._n_caps = n_caps
        self._n_caps_dims = n_caps_dims
        self._n_votes = n_votes
        self._capsule_kwargs = capsule_kwargs
        self.capsule = CapsuleLayer(self._n_caps, self._n_caps_dims,
                                    self._n_votes, noise_scale=4., **self._capsule_kwargs)
        self._dummy_vote = nn.Parameter(torch.rand(1, 1, self._n_votes, 6))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self._dummy_vote.data, std=0.05)

    def forward(self, h, x, presence=None):
        batch_size, n_input_points = x.size()[:2]
        vote_shape = [batch_size, self._n_caps, self._n_votes, 6]
        res = self.capsule(h)
        res.vote = res.vote[Ellipsis, :-1, :].view(vote_shape)
        votes, scale, vote_presence_prob = res.vote, res.scale, res.vote_presence

        # predicted pose (expand_x) from primary_encode should close to gaussian distributions (mean:votes, std:scale) calculated as follows
        expanded_x = x.unsqueeze(dim=1)
        gaussians = torch.distributions.Normal(votes, scale.unsqueeze(dim=-1))
        vote_log_prob_per_dim = gaussians.log_prob(expanded_x)
        vote_log_prob = vote_log_prob_per_dim.sum(dim=-1)

        dummy_vote_log_prob = torch.zeros([batch_size, 1, n_input_points])
        dummy_vote_log_prob -= 2. * torch.log(torch.tensor(10.))

        # vote_log_prob is the probabilty of obj-caps agreement
        # [B, n_caps+1， n_input_points]
        vote_log_prob = torch.cat([vote_log_prob, dummy_vote_log_prob.to(vote_log_prob.device)], 1)


        # mixing_logits is the probabilty of obj-caps agreement predicted by self._decoder module.
        # vote_log_prob should be consistent with mixing_logits.
        mixing_logits = safe_log(vote_presence_prob)

        dummy_logit = torch.zeros([batch_size, 1, n_input_points])
        dummy_logit -= 2. * torch.log(torch.tensor(10.))

        mixing_logits = torch.cat([mixing_logits, dummy_logit.to(mixing_logits.device)], 1)
        mixing_log_prob = mixing_logits - torch.logsumexp(mixing_logits, dim=1, keepdim=True)
        mixture_log_prob_per_point = torch.logsumexp(mixing_logits + vote_log_prob, dim=1) # here should be mixing_log_prob rather than mixing_logits mixing_log_prob = log$(a_k*a_(k,m))/(sumi(a_i)sumj(a_ij))$

        if presence is not None:
            mixture_log_prob_per_point = mixture_log_prob_per_point * presence.squeeze(dim=-1)
        mixture_log_prob_per_example = torch.sum(mixture_log_prob_per_point, 1)
        mixture_log_prob_per_batch = torch.mean(mixture_log_prob_per_example)

        posterior_mixing_logits_per_point = mixing_logits + vote_log_prob

        winning_vote_idx = torch.argmax(posterior_mixing_logits_per_point[:, :-1], 1)

        batch_idx = torch.arange(batch_size, dtype=torch.int64).unsqueeze(dim=-1)
        batch_idx = batch_idx.repeat(1, n_input_points).view(-1)

        point_idx = torch.arange(n_input_points, dtype=torch.int64).unsqueeze(dim=0)
        point_idx = point_idx.repeat(batch_size, 1).view(-1)

        # idx = torch.stack([batch_idx, winning_vote_idx.view(-1), point_idx], -1)
        # winning_vote = torch.gather(votes, idx)
        # winning_pres = torch.gather(vote_presence_prob, idx)
        winning_vote = votes[batch_idx, winning_vote_idx.view(-1), point_idx].view(batch_size, n_input_points, -1)
        winning_pres = vote_presence_prob[batch_idx, winning_vote_idx.view(-1), point_idx].view(batch_size, n_input_points)
        vote_presence = (mixing_logits[:, -1:] < mixing_logits[:, :-1]) ### if foreground is smaller than bg, ignore this pixel.

        # the first four votes belong to the square
        is_from_capsule = winning_vote_idx // n_input_points

        posterior_mixing_probs = torch.softmax(posterior_mixing_logits_per_point, 1)
        dummy_vote = self._dummy_vote.repeat(batch_size, 1, 1, 1)
        dummy_pres = torch.zeros([batch_size, 1, n_input_points])

        votes = torch.cat((votes, dummy_vote), dim=1)
        pres = torch.cat([vote_presence_prob, dummy_pres.to(vote_presence_prob.device)], dim=1)
        # pres = vote_presence_prob

        soft_winner = torch.sum(posterior_mixing_probs.unsqueeze(dim=-1) * votes, 1)
        soft_winner_pres = torch.sum(posterior_mixing_probs * pres, 1)

        posterior_mixing_probs = posterior_mixing_probs[:, :-1].permute(0, 2, 1) ##??
        # posterior_mixing_probs = posterior_mixing_probs.permute(0, 2, 1)

        caps_presence_prob = torch.max(vote_presence_prob, 2).values
        res.caps_presence_prob = caps_presence_prob

        assert winning_vote.shape == x.shape
        return res, self.OutputTuple(
        log_prob=mixture_log_prob_per_batch,
        vote_presence=vote_presence,
        winner=winning_vote,
        winner_pres=winning_pres,
        soft_winner=soft_winner,
        soft_winner_pres=soft_winner_pres,
        posterior_mixing_probs=posterior_mixing_probs,
        is_from_capsule=is_from_capsule,
        mixing_logits=mixing_logits,
        mixing_log_prob=mixing_log_prob,
    )
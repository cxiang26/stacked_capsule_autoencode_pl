import torch
from torch import nn

from .util import safe_log

class scae_loss(nn.Module):
    def __init__(self,
                 dynamic_l2_weight=10.,
                 caps_l1_weight=1.,
                 num_classes=10,
                 prior_within_example_sparsity_weight=1.,
                 prior_between_example_sparsity_weight=1.,
                 posterior_within_example_sparsity_weight=10.,
                 posterior_between_example_sparsity_weight=10.,
                 primary_caps_sparsity_weight=0.
                 ):
        super(scae_loss, self).__init__()
        self._dynamic_l2_weight = dynamic_l2_weight
        self._caps_l1_weight = caps_l1_weight
        self._num_classes = num_classes

        # self._prior_sparsity_loss_type = prior_sparsity_loss_type
        self._prior_within_example_sparsity_weight = prior_within_example_sparsity_weight
        self._prior_between_example_sparsity_weight = prior_between_example_sparsity_weight
        # self._posterior_sparsity_loss_type = poseterior_sparsity_loss_type
        self._posterior_within_example_sparsity_weight = posterior_within_example_sparsity_weight
        self._posterior_between_example_sparsity_weight = posterior_between_example_sparsity_weight
        # self._prior_within_example_constant = prior_within_example_constant
        self._primary_caps_sparsity_weight = primary_caps_sparsity_weight

    def forward(self, res):
        n_points = int(res.posterior_mixing_probs.shape[1])
        mass_explained_by_capsule = torch.sum(res.posterior_mixing_probs, 1)/n_points

        # entropy in capsule activations

        within_example = mass_explained_by_capsule / (torch.sum(mass_explained_by_capsule, 1, keepdim=True) + 1e-8)
        posterior_within_sparsity_loss = torch.mean(-torch.sum(within_example * safe_log(within_example), dim=-1))

        between_example = torch.sum(mass_explained_by_capsule, 0)
        between_example = between_example / (torch.sum(between_example, 0, keepdim=True) + 1e-8)
        posterior_between_sparsity_loss = torch.mean(-torch.sum(between_example * safe_log(between_example), dim=-1))

        # l2 penalty on capsule activations
        batch_size, num_caps = res.caps_presence_prob.size()
        within_example_constant = float(num_caps) / self._num_classes
        between_example_constant = float(batch_size) / self._num_classes
        prior_within_sparsity_loss = torch.sum((torch.sum(res.caps_presence_prob, 1) - within_example_constant) ** 2 / batch_size)
        prior_between_sparsity_loss = -torch.sum((torch.sum(res.caps_presence_prob, 0) - between_example_constant) ** 2 / num_caps)

        # all loss
        total_loss = (- res.rec_ll
                - self._caps_l1_weight * res.log_prob
                + self._dynamic_l2_weight * res.dynamic_weights_l2
                + self._primary_caps_sparsity_weight * res.primary_caps_l1
                + self._posterior_within_example_sparsity_weight * posterior_within_sparsity_loss
                - self._posterior_between_example_sparsity_weight * posterior_between_sparsity_loss
                + self._prior_within_example_sparsity_weight * prior_within_sparsity_loss
                - self._prior_between_example_sparsity_weight * prior_between_sparsity_loss
                )
        return total_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch.distributions as dist

import logging
logger = logging.getLogger(__name__)


class NoisyLinear(nn.Module):

  def __init__(self, in_features, out_features):
    super(NoisyLinear, self).__init__()

    self.in_features = in_features
    self.out_features = out_features

    self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
    self.sigma_w = nn.Parameter(torch.Tensor(out_features, in_features))

    self.mu_b = nn.Parameter(torch.Tensor(out_features))
    self.sigma_b = nn.Parameter(torch.Tensor(out_features))

    self.epsilon_out = torch.Tensor(out_features)
    self.epsilon_in = torch.Tensor(in_features)

    self.in_dist = dist.normal.Normal(0, 1)
    self.out_dist = dist.normal.Normal(0, 1)

    self.sample_epsilon()
    self._reset_parameters()

  def sample_epsilon(self):
    with torch.no_grad():
      self.epsilon_out = self.out_dist.sample(torch.Size([self.out_features])).to('cuda')
      self.epsilon_in = self.in_dist.sample(torch.Size([self.in_features])).to('cuda')

  def _reset_parameters(self):
    bound = 1 / np.sqrt(self.in_features)
    init.uniform_(self.mu_w, -bound, bound)
    init.uniform_(self.mu_b, -bound, bound)

    sigma_0 = 0.5
    constant = sigma_0 / np.sqrt(self.in_features)
    init.constant_(self.sigma_w, constant)
    init.constant_(self.sigma_b, constant)

  @staticmethod
  def _f(x):
    return torch.sign(x) * torch.sqrt(torch.abs(x))


  def forward(self, input):
    with torch.no_grad():
      epsilon_b = self._f(self.epsilon_out)
      epsilon_w = epsilon_b.view(-1, 1) * self._f(self.epsilon_in).view(1, -1)
    w = self.mu_w + self.sigma_w * epsilon_w
    b = self.mu_b + self.sigma_b * epsilon_b
    return F.linear(input, w, b)

  def extra_repr(self):
    s = f'in_features={self.in_features} out_features={self.out_features}, bias=True'
    return s
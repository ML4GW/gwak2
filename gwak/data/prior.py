from collections import OrderedDict

import torch
from torch.distributions.uniform import Uniform

from ml4gw.distributions import Cosine


class Prior():

    def __init__(self, **kwargs):

        self.params = OrderedDict()
        for k, v in kwargs.items():
            self.params[k] = v

    def sample(self, batch_size):

        self.sampled_params = OrderedDict()

        for k in self.params.keys():
            self.sampled_params[k] = self.params[k].sample((batch_size,))

        return self.sampled_params


class SineGaussian:

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        self.intrinsic_prior = Prior(
            hrss = Uniform(1e-21, 2e-21),
            quality = Uniform(5, 75),
            frequency = Uniform(64, 512),
            phase = Uniform(0, 2 * torch.pi),
            eccentricity = Uniform(0, 0.01),
        )

        self.extrinsic_prior = Prior(
            ra = Uniform(0, 2 * torch.pi),
            dec = Cosine(),
            psi = Uniform(0, 2 * torch.pi)
        )

    def intrinsic_prior(self):
        return self.intrinsic_prior

    def extrinsic_prior(self):
        return self.extrinsic_prior
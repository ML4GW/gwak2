import logging
from collections import OrderedDict

import torch
from torch.distributions.uniform import Uniform

from ml4gw.distributions import Cosine, Sine
from bilby.gw.conversion import transform_precessing_spins


class Constant:

    def __init__(self, val, tensor=True):
        self.val = val
        self.tensor = tensor

    def __repr__(cls):
        return self.__name__

    def sample(self, batch_size):

        if self.tensor:
            return torch.full(batch_size, self.val)

        return self.val


class BasePrior:

    def __init__(self):
        self.params = OrderedDict()
        self.sampled_params = OrderedDict()

    def sample(self, batch_size):

        self.sampled_params = OrderedDict()

        for k in self.params.keys():

            if type(self.params[k]) == Constant:
                    self.sampled_params[k] = self.params[k].val
            else:
                self.sampled_params[k] = self.params[k].sample((batch_size,))

        return self.sampled_params


class SineGaussianHighFrequency(BasePrior):

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        super().__init__()
        self.params = OrderedDict(
            hrss = Uniform(1e-21, 2e-21),
            quality = Uniform(5, 75),
            frequency = Uniform(512, 1024),
            phase = Uniform(0, 2 * torch.pi),
            eccentricity = Uniform(0, 0.01),
            ra = Uniform(0, 2 * torch.pi),
            dec = Cosine(),
            psi = Uniform(0, 2 * torch.pi)
        )


class SineGaussianLowFrequency(BasePrior):

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        super().__init__()
        self.params = OrderedDict(
            hrss = Uniform(1e-21, 2e-21),
            quality = Uniform(5, 75),
            frequency = Uniform(64, 512),
            phase = Uniform(0, 2 * torch.pi),
            eccentricity = Uniform(0, 0.01),
            ra = Uniform(0, 2 * torch.pi),
            dec = Cosine(),
            psi = Uniform(0, 2 * torch.pi)
        )


class BBHPrior(BasePrior):

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        super().__init__()
        # taken from bilby.gw.prior.BBHPriorDict()
        self.params = dict(
            # mass_1 and mass_2 used to be `Constraint()`
            mass_1 = Uniform(5, 100),
            mass_2 = Uniform(5, 100),
            mass_ratio = Uniform(0.125, 1),
            chirp_mass = Uniform(25, 100),
            theta_jn = Sine(),
            phase = Uniform(0, 2 * torch.pi),
            a_1 = Uniform(0, 0.99),
            a_2 = Uniform(0, 0.99),
            tilt_1 = Sine(0, torch.pi),
            tilt_2 = Sine(0, torch.pi),
            phi_12 = Uniform(0, 2 * torch.pi),
            phi_jl = Uniform(0, 2 * torch.pi),
            reference_frequency = Constant(50.0, tensor=False),
            # CHECK THIS: time of coallesence and fs
            tc = Constant(0),
            fs = Constant(2048),
            dist_mpc = Uniform(50, 200),
            ra = Uniform(0, 2 * torch.pi),
            dec = Cosine(),
            psi = Uniform(0, torch.pi)
        )

    def sample(self, batch_size):

        for k in self.params.keys():
            self.sampled_params[k] = self.params[k].sample((batch_size,))

        # convert from Bilby convention to Lalsimulation
        self.sampled_params['incl'], self.sampled_params['s1x'], self.sampled_params['s1y'], \
        self.sampled_params['s1z'], self.sampled_params['s2x'], self.sampled_params['s2y'], \
        self.sampled_params['s2z'] = transform_precessing_spins(
            self.sampled_params['theta_jn'], self.sampled_params['phi_jl'],
            self.sampled_params['tilt_1'],
            self.sampled_params['tilt_2'], self.sampled_params['phi_12'],
            self.sampled_params['a_1'], self.sampled_params['a_2'],
            self.sampled_params['mass_1'], self.sampled_params['mass_2'],
            self.sampled_params['reference_frequency'], self.sampled_params['phase']
            )

        self.sampled_params['incl'] = torch.Tensor(self.sampled_params['incl'])
        self.sampled_params['s1x'] = torch.Tensor(self.sampled_params['s1x'])
        self.sampled_params['s1y'] = torch.Tensor(self.sampled_params['s1y'])
        self.sampled_params['s1z'] = torch.Tensor(self.sampled_params['s1z'])
        self.sampled_params['s2x'] = torch.Tensor(self.sampled_params['s2x'])
        self.sampled_params['s2y'] = torch.Tensor(self.sampled_params['s2y'])
        self.sampled_params['s2z'] = torch.Tensor(self.sampled_params['s2z'])

        self.sampled_params['f_ref'] = self.sampled_params['reference_frequency']
        self.sampled_params['phiRef'] = self.sampled_params['phase']

        return self.sampled_params

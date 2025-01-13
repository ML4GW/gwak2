import logging
from collections import OrderedDict

import torch
from torch.distributions.uniform import Uniform

import lal
from astropy import units as u
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
            mass_ratio = Uniform(0.5, 0.99), # Uniform(0.125, 1),
            chirp_mass = Uniform(15, 30), # Uniform(25, 100),
            theta_jn = Sine(),
            phase = Constant(0), # Uniform(0, 2 * torch.pi),
            a_1 = Uniform(0, 0.99),
            a_2 = Uniform(0, 0.99),
            tilt_1 = Sine(0, torch.pi),
            tilt_2 = Sine(0, torch.pi),
            phi_12 = Uniform(0, 2 * torch.pi),
            phi_jl = Uniform(0, 2 * torch.pi),
            reference_frequency = Constant(20.0, tensor=False), #Constant(50.0, tensor=False),
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

        self.sampled_params['mass_2'] = self.sampled_params['chirp_mass'] * (1 + self.sampled_params['mass_ratio']) ** 0.2 / self.sampled_params['mass_ratio']**0.6
        self.sampled_params['mass_1'] = self.sampled_params['mass_ratio'] * self.sampled_params['mass_2']

        # if self.sampled_params['mass_2'] > self.sampled_params['mass_1']:
        #     self.sampled_params['mass_1'], self.sampled_params['mass_2'] = self.sampled_params['mass_2'], self.sampled_params['mass_1']
        #     self.sampled_params['mass_ratio'] = 1 / self.sampled_params['mass_ratio']


        # # correct units
        # self.sampled_params['mass_2'] *= lal.MSUN_SI
        # self.sampled_params['mass_1'] *= lal.MSUN_SI

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
        self.sampled_params['s1x'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s1x'])
        self.sampled_params['s1y'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s1y'])
        self.sampled_params['s1z'] = torch.Tensor(self.sampled_params['s1z'])
        self.sampled_params['s2x'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s2x'])
        self.sampled_params['s2y'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s2y'])
        self.sampled_params['s2z'] = torch.Tensor(self.sampled_params['s2z'])

        self.sampled_params['f_ref'] = self.sampled_params['reference_frequency']
        self.sampled_params['phiRef'] = self.sampled_params['phase']

        self.sampled_params['dist_mpc'] = (self.sampled_params['dist_mpc'] * u.Mpc).to("m").value # ???

        logger = logging.getLogger(__name__)

        for k in self.sampled_params.keys():
            if type(self.sampled_params[k])==float:
                logger.info(f'The shape of {k} is {self.sampled_params[k]}')
            else:
                logger.info(f'The shape of {k} is {self.sampled_params[k].shape}')

        return self.sampled_params

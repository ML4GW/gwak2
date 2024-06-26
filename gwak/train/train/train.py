import numpy as np
from typing import Callable, List, Optional

import torch
from torch.distributions.uniform import Uniform

import ml4gw
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.waveforms import SineGaussian
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.distributions import Cosine, PowerLaw
from ml4gw.gw import compute_observed_strain, get_ifo_geometry
from pathlib import Path

import gwak
from gwak.data.prior import Prior

from models import LSTM_AE_SPLIT


def main(
    data_dir: str = '/home/ethan.marx/amplfi/data/train/background',
    ifos: List = ['H1', 'L1'],
    sample_rate: int = 4096,
    psd_length: int = 64, # segment length in seconds to estimate PSD
    batch_size: int = 1,
    kernel_length: float = 200/4096, # number of sec passed in,
    fftlength: float = 2,
    fduration: float = 1, # whitening parameter: how much data to cut off,
):

    data_dir = Path(data_dir)
    print(f'Running training on {data_dir}')

    kernel_size = (psd_length + fduration + kernel_length) * sample_rate

    # create training dataloader
    dataloader = Hdf5TimeSeriesDataset(
        fnames = list(data_dir.glob('*.hdf5')),
        channels = ifos,
        kernel_size = int(kernel_size),
        batch_size = batch_size,
        batches_per_epoch = 200,
        coincident = False
    )

    # psd estimator
    # takes tensor of shape (batch_size, num_ifos, psd_length)
    spectral_density = SpectralDensity(
        sample_rate,
        fftlength,
        average = "median"
    )

    # create whitener
    whitener = Whiten(
        fduration,
        sample_rate,
        highpass = 30,
    )

    sine_gaussian = SineGaussian(sample_rate, duration=kernel_length + fduration)

    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
    intrinsic_prior = Prior(
      hrss = Uniform(1e-21, 2e-21),
      quality = Uniform(5, 75),
      frequency = Uniform(64, 512),
      phase = Uniform(0, 2 * torch.pi),
      eccentricity = Uniform(0, 0.01),
    )

    extrinsic_prior = Prior(
      ra = Uniform(0, 2 * torch.pi),
      dec = Cosine(),
      psi = Uniform(0, 2 * torch.pi)
    )

    model = LSTM_AE_SPLIT(len(ifos), 200, 4)

    for X in dataloader:
      # X is shape (batch_size, num_ifos, kernel_size)

      # split X into psd data and data to be whitened
      split_size = int((kernel_length + fduration) * sample_rate)
      splits = [X.size(-1) - split_size, split_size]
      psd_data, X = torch.split(X, splits, dim=-1)

      # calculate psds
      psds = spectral_density(psd_data)


      # sample from prior and generate waveforms
      parameters = intrinsic_prior.sample(batch_size) # dict[str, torch.tensor]
      cross, plus = sine_gaussian(**parameters)

      # sample extrinsic parameters
      ra, dec, psi = extrinsic_prior.sample(batch_size)

      # get detector orientations
      tensors, vertices = get_ifo_geometry(*ifos)

      # compute detector responses
      responses = compute_observed_strain(
          dec,
          psi,
          ra,
          tensors,
          vertices,
          sample_rate,
          cross=cross,
          plus=plus
      )

      # inject into data and whiten
      injected = X + responses
      whitened = whitener(X, psds)

      loss = model(whitened)
      loss.backward()


if __name__=='__main__':
    main()
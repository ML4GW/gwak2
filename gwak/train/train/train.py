import argparse
import numpy as np
from typing import Callable, List, Optional

import torch
import torch.optim as optim
from torch.distributions.uniform import Uniform

import ml4gw
from ml4gw.dataloading import Hdf5TimeSeriesDataset
import ml4gw.waveforms as waveforms # import SineGaussian
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.distributions import Cosine, PowerLaw
from ml4gw.gw import compute_observed_strain, get_ifo_geometry
from pathlib import Path

import gwak
from gwak.data import Prior

import models


def main(
    data: torch.nn.Module,
    model_name: str,
    data_dir: str = '/home/ethan.marx/aframe/data/train/background',
    ifos: List = ['H1', 'L1'],
    sample_rate: int = 2048,
    psd_length: int = 64, # segment length in seconds to estimate PSD
    batch_size: int = 128,
    kernel_length: float = 200/2048, # number of sec passed in,
    fftlength: float = 2,
    fduration: float = 1, # whitening parameter: how much data to cut off,
):

    data_dir = Path(data_dir)
    print(f'Running training on {data_dir}')

    kernel_size = (psd_length + fduration + kernel_length) * sample_rate
    batches_per_epoch = 128

    # create training dataloader
    dataloader = Hdf5TimeSeriesDataset(
        fnames = list(data_dir.glob('*.hdf5')),
        channels = ifos,
        kernel_size = int(kernel_size),
        batch_size = batch_size,
        batches_per_epoch = batches_per_epoch, # ??
        coincident = False
    )

    num_workers = 5
    dataloader = torch.utils.data.DataLoader(
        dataloader, num_workers=num_workers, pin_memory=False
    )

    # psd estimator
    # takes tensor of shape (batch_size, num_ifos, psd_length)
    spectral_density = SpectralDensity(
        sample_rate,
        fftlength,
        average = 'median'
    ).to('cuda')

    # create whitener
    whitener = Whiten(
        fduration,
        sample_rate,
        highpass = 30,
    ).to('cuda')

    data = getattr(waveforms, data)(sample_rate, duration=kernel_length + fduration)

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

    model_class = getattr(models, model_name)
    model = model_class(len(ifos), 200, 8).to('cuda')

    loss_fn = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    training_history = {
        'train_loss': []    }

    # training loop
    epoch_count = 0
    EPOCHS = 20

    for epoch_num in range(EPOCHS):
        epoch_count += 1
        epoch_train_loss = 0
        print(f'Epoch {epoch_count}')
        for X in dataloader:
            X = X[0].to('cuda')
            optimizer.zero_grad()
            # X is shape (batch_size, num_ifos, kernel_size)

            # split X into psd data and data to be whitened
            split_size = int((kernel_length + fduration) * sample_rate)
            splits = [X.size(-1) - split_size, split_size]
            psd_data, X = torch.split(X, splits, dim=-1)

            # calculate psds
            psds = spectral_density(psd_data.double())

            # sample from prior and generate waveforms
            parameters = intrinsic_prior.sample(batch_size) # dict[str, torch.tensor]
            cross, plus = data(**parameters)

            # sample extrinsic parameters
            ra, dec, psi = extrinsic_prior.sample(batch_size).values()

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
              cross=cross.float(),
              plus=plus.float()
            ).to('cuda')

            # inject into data and whiten
            injected = X + responses

            whitened = whitener(injected.double(), psds.double())

            # normalize the input data
            stds = torch.std(whitened, dim=-1, keepdim=True)
            whitened = whitened / stds

            response = model(whitened)
            loss = loss_fn(model(whitened), whitened)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= batches_per_epoch
        print(f'Epoch loss is {epoch_train_loss}')
        training_history['train_loss'].append(epoch_train_loss)

    # save the model
    torch.save(model.state_dict(), f'model.pt')

    # plot training history
    from matplotlib import pyplot as plt
    epochs = np.linspace(1, epoch_count, epoch_count)
    plt.plot(epochs, np.array(training_history[
            'train_loss']), label='Training loss')
    plt.legend()
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid()
    plt.savefig('training_loss.pdf')
    plt.clf()

    plt.plot(model(whitened)[0,0,:].cpu().detach().numpy(), label='reco')
    plt.plot(whitened[0,0,:].cpu().detach().numpy(), label='reco')
    plt.legend()
    plt.grid()
    plt.savefig('reconstruction.pdf')
    plt.clf()

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--data', type=str, help='Directory containing the injections to evaluate')
    parser.add_argument('--model', type=str,
                        help='Directory containing the injections to evaluate')

    args = parser.parse_args()

    main(args.data, args.model)
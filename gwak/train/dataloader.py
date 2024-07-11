import argparse
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional

import wandb
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import ml4gw
from ml4gw.dataloading import Hdf5TimeSeriesDataset
import ml4gw.waveforms as waveforms
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.gw import compute_observed_strain, get_ifo_geometry

import models
import dataloader
from gwak.data import prior


class GwakDataloader(pl.LightningDataModule):

    def __init__(
        self,
        data_dir='/home/ethan.marx/aframe/data/train/background',
        sample_rate=2048,
        kernel_length=200/2048,
        psd_length=64,
        fduration=1,
        fftlength=2,
        batch_size=128,
        batches_per_epoch=128,
        num_workers=5,
    ):
        super().__init__()
        self.train_fnames, self.val_fnames = self.train_val_split(data_dir)
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.psd_length = psd_length
        self.fduration = fduration
        self.fftlength = fftlength
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.num_workers = num_workers


    def train_val_split(self, data_dir, val_split=0.2):

        all_files = list(Path(data_dir).glob('*.hdf5'))
        n_all_files = len(all_files)
        n_train_files = int(n_all_files * (1 - val_split))

        return all_files[:n_train_files], all_files[n_train_files:]

    def generate_waveforms(self, batch_size):
        # data generation
        data = waveforms.SineGaussian(self.sample_rate, duration=self.kernel_length + self.fduration)


        # priors
        signal_prior = prior.SineGaussian()

        intrinsic_prior = signal_prior.intrinsic_prior
        extrinsic_prior = signal_prior.extrinsic_prior

        # sample extrinsic parameters
        ra, dec, psi = extrinsic_prior.sample(batch_size).values()

        # get detector orientations
        ifos = ['H1', 'L1']
        tensors, vertices = get_ifo_geometry(*ifos)

        # sample from prior and generate waveforms
        parameters = intrinsic_prior.sample(batch_size) # dict[str, torch.tensor]
        cross, plus = data(**parameters)

        # compute detector responses
        responses = compute_observed_strain(
          dec,
          psi,
          ra,
          tensors,
          vertices,
          self.sample_rate,
          cross=cross.float(),
          plus=plus.float()
        ).to('cuda')

        return responses

    def inject(self, batch, waveforms):

        # split batch into psd data and data to be whitened
        split_size = int((self.kernel_length + self.fduration) * self.sample_rate)
        splits = [batch.size(-1) - split_size, split_size]
        psd_data, batch = torch.split(batch, splits, dim=-1)

        # psd estimator
        # takes tensor of shape (batch_size, num_ifos, psd_length)
        spectral_density = SpectralDensity(
            self.sample_rate,
            self.fftlength,
            average = 'median'
        ).to('cuda')

        # calculate psds
        psds = spectral_density(psd_data.double())

        # inject into data and whiten
        injected = batch + waveforms


        # create whitener
        whitener = Whiten(
            self.fduration,
            self.sample_rate,
            highpass = 30,
        ).to('cuda')

        whitened = whitener(injected.double(), psds.double())

        # normalize the input data
        stds = torch.std(whitened, dim=-1, keepdim=True)
        whitened = whitened / stds

        return whitened

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training or self.trainer.validating or self.trainer.sanity_checking:
            # unpack the batch
            [batch] = batch
            # generate waveforms
            waveforms = self.generate_waveforms(batch.shape[0])
            # inject waveforms; maybe also whiten data preprocess etc..
            batch = self.inject(batch, waveforms)

            return batch

    def val_dataloader(self):
        dataset = Hdf5TimeSeriesDataset(
            self.val_fnames,
            channels=['H1', 'L1'],
            kernel_size=int((self.psd_length + self.fduration + self.kernel_length) * self.sample_rate), # int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
        )

        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=False
        )
        return dataloader

    def train_dataloader(self):

        dataset = Hdf5TimeSeriesDataset(
                self.train_fnames,
                channels=['H1', 'L1'],
                kernel_size=int((self.psd_length + self.fduration + self.kernel_length) * self.sample_rate),#int(self.sample_rate * self.sample_length),
                batch_size=self.batch_size,
                batches_per_epoch=self.batches_per_epoch,
                coincident=False,
            )

        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=False
        )
        return dataloader
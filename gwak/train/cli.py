import argparse
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional

import wandb
import torch
from lightning.pytorch.cli import LightningCLI

import ml4gw
from ml4gw.dataloading import Hdf5TimeSeriesDataset
import ml4gw.waveforms as waveforms
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.gw import compute_observed_strain, get_ifo_geometry

import models
from dataloader import GwakDataloader
from gwak.data import prior


def cli_main():

    cli = LightningCLI(models.Autoencoder, GwakDataloader)

if __name__ == "__main__":
    cli_main()
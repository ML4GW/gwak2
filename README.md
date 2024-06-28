# GWAK 2.0 🥑🦾

This repo is dedicated to the updated version of the algorithm presented in the [MLST](https://iopscience.iop.org/article/10.1088/2632-2153/ad3a31). 

![DALL·E 2024-06-07 15 21 58 - A futuristic and artistic version of an avocado  The avocado is designed with sleek, metallic textures and glowing neon pink and yellow accents  The s](https://github.com/ML4GW/gwak2/assets/4249113/f396688b-125e-48f1-bbd5-48f3c9854e8e)

The current projects include
- [`data`](./gwak/data/README.md) - Scripts for generating training and testing data
- [`train`](./gwak/train/README.md) - Pytorch (lightning) code for training neural-networks


The project uses `poetry`, `Conda` and `Snakemake` to run the code. Follow installation instructions below to prepare your environment.


## Installation ⚙️


### Optional step (only if you don't have `Miniconda3`)
If you **do not have** `Miniconda` installed on your machine, follow first those steps
- use the [`quickstart`](https://github.com/ML4GW/quickstart) repo to setup `Miniconda` and install `poetry`
```
$ git clone git@github.com:ml4gw/quickstart.git
$ cd quickstart
$ make
```
- if you see this error
```
Verifying checksum... Done.
Preparing to install helm into /you/path/miniconda3-tmp/bin/
helm installed into /you/path/miniconda3-tmp/bin//helm
helm not found. Is /you/path/miniconda3-tmp/bin/ on your $PATH?
Failed to install helm
	For support, go to https://github.com/helm/helm.
make: *** [Makefile:65: install-helm] Error 1
```
- do the following commands
```
$ source ~/.bashrc
$ make install-poetry install-kubectl install-s3cmd
```
- if everything was installed successfully, continue to the steps below


### Main installation

If you **do have** `Miniconda` already installed on your machine, follow those steps
- checkout this repo
```
$ git clone git@github.com:ML4GW/gwak2.git
& cd gwak2
```
- clone submodules (such as `ml4gw`)
```
git submodule update --init --recursive
```
- create a new `Conda` environment
```
$ conda env create -n gwak --file environment.yaml
$ conda activate gwak
```
- install `gwak` project in editing mode
```
$ pip install -e .
```

Now you are ready to *gwak*!
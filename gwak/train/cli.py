from lightning.pytorch.cli import LightningCLI

import models
import dataloader

def cli_main():

    cli = LightningCLI(save_config_kwargs={"overwrite": True})
    # cli = LightningCLI(models.Autoencoder, dataloader.GwakDataloader)


if __name__ == "__main__":
    cli_main()
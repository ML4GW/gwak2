import os
import time
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning.pytorch as pl


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path, arch=pl_module.model, metric=pl_module.metric
        )

        save_dir = trainer.logger.log_dir or trainer.logger.save_dir
        torch.save(module.model.state_dict(), os.path.join(save_dir, 'model.pt'))


class LargeLinear(pl.LightningModule):

    def __init__(self, num_ifos, num_timesteps, bottleneck):
        super(LargeLinear, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.Linear1 = nn.Linear(num_timesteps * 2, 2**7)
        self.Linear2 = nn.Linear(2**7, 2**9)
        self.Linear3 = nn.Linear(2**9, bottleneck)
        self.Linear4 = nn.Linear(bottleneck, 2**9)
        self.Linear5 = nn.Linear(2**9, 2**7)
        self.Linear6 = nn.Linear(2**7, num_timesteps * 2)

    def training_step(self, batch, batch_idx):

        x, _ = batch
        batch_size = x.shape[0]

        x = x.reshape(-1, self.num_timesteps * self.num_ifos)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        x = F.relu(self.Linear4(x))
        x = F.tanh(self.Linear5(x))
        x = (self.Linear6(x))
        x = x.reshape(batch_size, self.num_ifos, self.num_timesteps)

        loss_fn = torch.nn.L1Loss()

        loss = loss_fn(rec, x)

        self.log("train_loss", loss)
        return loss


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim // 2
        self.rnn1_0 = nn.LSTM(
            input_size=1,
            hidden_size=4,
            num_layers=2,
            batch_first=True
        )
        self.rnn1_1 = nn.LSTM(
            input_size=1,
            hidden_size=4,
            num_layers=2,
            batch_first=True
        )

        self.encoder_dense_scale = 20
        self.linear1 = nn.Linear(
            in_features=2**8, out_features=self.encoder_dense_scale * 4)
        self.linear2 = nn.Linear(
            in_features=self.encoder_dense_scale * 4, out_features=self.encoder_dense_scale * 2)
        self.linear_passthrough = nn.Linear(
            2 * seq_len, self.encoder_dense_scale * 2)
        self.linear3 = nn.Linear(
            in_features=self.encoder_dense_scale * 4, out_features=self.embedding_dim)

        self.linearH = nn.Linear(4 * seq_len, 2**7)
        self.linearL = nn.Linear(4 * seq_len, 2**7)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, 2 * self.seq_len)
        other_dat = self.linear_passthrough(x_flat)
        Hx, Lx = x[:, :, 0][:, :, None], x[:, :, 1][:, :, None]

        Hx, (_, _) = self.rnn1_0(Hx)
        Hx = Hx.reshape(batch_size, 4 * self.seq_len)
        Hx = F.tanh(self.linearH(Hx))

        Lx, (_, _) = self.rnn1_1(Lx)
        Lx = Lx.reshape(batch_size, 4 * self.seq_len)
        Lx = F.tanh(self.linearL(Lx))

        x = torch.cat([Hx, Lx], dim=1)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = torch.cat([x, other_dat], axis=1)
        x = F.tanh(self.linear3(x))

        return x.reshape((batch_size, self.embedding_dim))  # phil harris way


class Decoder(nn.Module):

    def __init__(self, seq_len, n_features=1, input_dim=64,):
        super().__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.rnn1_0 = nn.LSTM(
            input_size=1,
            hidden_size=1,
            num_layers=1,
            batch_first=True
        )
        self.rnn1_1 = nn.LSTM(
            input_size=1,
            hidden_size=1,
            num_layers=1,
            batch_first=True
        )
        self.rnn1 = nn.LSTM(
            input_size=2,
            hidden_size=2,
            num_layers=1,
            batch_first=True
        )

        self.linearH = nn.Linear(2 * self.seq_len, self.seq_len)
        self.linearL = nn.Linear(2 * self.seq_len, self.seq_len)

        self.linear1 = nn.Linear(self.hidden_dim, 2**8)
        self.linear2 = nn.Linear(2**8, 2 * self.seq_len)

    def forward(self, x):

        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        Hx = self.linearH(x)[:, :, None]
        Lx = self.linearL(x)[:, :, None]

        x = torch.cat([Hx, Lx], dim=2)

        return x


class Autoencoder(pl.LightningModule):

    def __init__(
        self,
        num_ifos: int = 2,
        num_timesteps: int = 200,
        bottleneck: int = 8
        ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.bottleneck = bottleneck
        self.model = Encoder(
            seq_len=num_timesteps, n_features=num_ifos, embedding_dim=bottleneck)
        self.decoder = Decoder(
            seq_len=num_timesteps, n_features=num_ifos, input_dim=bottleneck)

    def training_step(self, batch, batch_idx):
        x = batch

        x = x.transpose(1, 2)
        x = self.model(x)
        x = self.decoder(x)

        x = x.transpose(1, 2)

        loss_fn = torch.nn.L1Loss()

        self.metric = loss_fn(batch, x)

        self.log(
            'train_loss',
            self.metric,
            on_epoch=True,
            )

        return self.metric

    def validation_step(self, batch, batch_idx):
        x = batch

        x = x.transpose(1, 2)
        x = self.model(x)
        x = self.decoder(x)

        x = x.transpose(1, 2)

        loss_fn = torch.nn.L1Loss()

        loss = loss_fn(batch, x)

        self.log(
            'val_loss',
            loss,
            on_epoch=True,
            sync_dist=True
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        # checkpoint for saving best model
        # that will be used for downstream export
        # and inference tasks
        # checkpoint for saving multiple best models
        callbacks = []

        # if using ray tune don't append lightning
        # model checkpoint since we'll be using ray's
        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            save_last=True,
            auto_insert_metric_name=False
            )

        callbacks.append(checkpoint)

        return callbacks


class gwak2(pl.LightningModule):

    def __init__(self, ):

        super().__init__()

    def training_step(self, batch, batch_idx):
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        return optimizer

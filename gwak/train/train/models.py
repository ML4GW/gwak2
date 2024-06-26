import torch
import torch.nn as nn
import torch.nn.functional as F


SEG_NUM_TIMESTEPS = 200
FACTORS_NOT_USED_FOR_FM = [3,7,11,15,19]
VERSION = 'v0'


class LinearModel(nn.Module):

    def __init__(self, n_dims):
        super(LinearModel, self).__init__()
        self.layer = nn.Linear(21-len(FACTORS_NOT_USED_FOR_FM), 1)

        if '_non_linear_bbh_only' in VERSION:

            self.layer1 = nn.Linear(21-len(FACTORS_NOT_USED_FOR_FM)-1, 32)
            self.layer2 = nn.Linear(32, 32)
            self.layer3 = nn.Linear(32, 32)
            self.layer4 = nn.Linear(32, 1)

    def forward(self, x):

        if '_non_linear_bbh_only' in VERSION:
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            return self.layer4(x)
        else:
            return self.layer(x)

class GwakClassifier(nn.Module):

    def __init__(self, n_dims):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(21-len(FACTORS_NOT_USED_FOR_FM), 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 1)


    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class FAT(nn.Module):

    def __init__(self, num_ifos, num_timesteps, BOTTLENECK):
        super(FAT, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.Linear1 = nn.Linear(num_timesteps * 2, 2**7)
        self.Linear2 = nn.Linear(2**7, 2**9)
        self.Linear3 = nn.Linear(2**9, BOTTLENECK)
        self.Linear4 = nn.Linear(BOTTLENECK, 2**9)
        self.Linear5 = nn.Linear(2**9, 2**7)
        self.Linear6 = nn.Linear(2**7, num_timesteps * 2)

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.reshape(-1, self.num_timesteps * self.num_ifos)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        x = F.relu(self.Linear4(x))
        x = F.tanh(self.Linear5(x))
        x = (self.Linear6(x))
        x = x.reshape(batch_size, self.num_ifos, self.num_timesteps)

        return x


class DUMMY_CNN_AE(nn.Module):

    def __init__(self, num_ifos, num_timesteps, BOTTLENECK):
        super(DUMMY_CNN_AE, self).__init__()
        print("WARNING: Change this with Eric's actual LSTM model!")
        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.Conv1 = nn.Conv1d(in_channels=num_ifos,
                               out_channels=5, kernel_size=5, padding='same')
        self.Linear1 = nn.Linear(num_timesteps * 5, SEG_NUM_TIMESTEPS)
        self.Linear2 = nn.Linear(SEG_NUM_TIMESTEPS, BOTTLENECK)
        self.Linear3 = nn.Linear(BOTTLENECK, num_timesteps * 5)
        self.Conv2 = nn.Conv1d(in_channels=5, out_channels=2,
                               kernel_size=5, padding='same')

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.Conv1(x))
        x = x.view(-1, self.num_timesteps * 5)
        x = F.tanh(self.Linear1(x))
        x = F.tanh(self.Linear2(x))
        x = F.tanh(self.Linear3(x))
        x = x.view(batch_size, 5, self.num_timesteps)
        x = self.Conv2(x)
        return x


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 4 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        self.linear0 = nn.Linear(
            in_features=self.embedding_dim * seq_len, out_features=self.hidden_dim * 10)
        self.linear1 = nn.Linear(
            in_features=self.hidden_dim * 10, out_features=self.hidden_dim * 5)
        self.linear2 = nn.Linear(
            in_features=self.hidden_dim * 5, out_features=self.hidden_dim * 2)
        self.linear3 = nn.Linear(
            in_features=self.hidden_dim * 2, out_features=self.embedding_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x, (hidden_n, cell_n) = self.rnn1(x)

        x, (hidden_n, cell_n) = self.rnn2(x)
        x = self.linear0(x.reshape(batch_size, -1))
        x = F.tanh(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        return x.reshape((batch_size, self.embedding_dim))  # phil harris way


class Decoder(nn.Module):

    def __init__(self, seq_len, n_features=1, input_dim=64,):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.linear1 = nn.Linear(self.hidden_dim, 2**9)
        self.linear2 = nn.Linear(2**9, self.hidden_dim * self.seq_len)

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]

        x = F.relu(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = x.reshape(batch_size, self.seq_len, self.hidden_dim)

        x, (hidden_n, cell_n) = self.rnn2(x)

        return self.output_layer(x)


class LSTM_AE(nn.Module):

    def __init__(self, num_ifos, num_timesteps, BOTTLENECK):
        super(LSTM_AE, self).__init__()
        print("WARNING: This is LITERALLY Eric's model!!!")
        for i in range(50):
            continue
            print("MINECRAFT MINECRAFT MINECRAFT")
        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.BOTTLENECK = BOTTLENECK
        self.encoder = Encoder(seq_len=num_timesteps,
                               n_features=num_ifos, embedding_dim=BOTTLENECK)
        self.decoder = Decoder(seq_len=num_timesteps,
                               n_features=num_ifos, input_dim=BOTTLENECK)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.transpose(1, 2)
        return x


# model which splits and uses separate LSTMs for each detecor channel
# torch.manual_seed(42)
def LSTM_N_params(hid, inp):
    return 4 * hid * inp + 4 * hid * hid + 2 * 4 * hid


class Encoder_SPLIT(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder_SPLIT, self).__init__()
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


class Encoder_SPLIT_intermediate(nn.Module):

    def __init__(self, seq_len, n_features, intermediate_position, batch_size=None, embedding_dim=64):
        super(Encoder_SPLIT_intermediate, self).__init__()
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

        self.intermediate_position = intermediate_position
        self.batch_size = batch_size

    def forward(self, x):
        if self.intermediate_position == "return_rnn":
            batch_size = x.shape[0]
            
            Hx, Lx = x[:, :, 0][:, :, None], x[:, :, 1][:, :, None]

            #t255 = time.time()
            Hx, (_, _) = self.rnn1_0(Hx)
            Hx = Hx.reshape(batch_size, 4 * self.seq_len)
            Hx = F.tanh(self.linearH(Hx))

            Lx, (_, _) = self.rnn1_1(Lx)
            Lx = Lx.reshape(batch_size, 4 * self.seq_len)
            Lx = F.tanh(self.linearL(Lx))

            x = torch.cat([Hx, Lx], dim=1)
            return x, batch_size
        
        elif self.intermediate_position == "return_embedding":
            x, orig = x
            orig = torch.swapaxes(orig, 1, 2)

            x_flat = orig.reshape(self.batch_size, 2 * self.seq_len)
            other_dat = self.linear_passthrough(x_flat)
            
            x = F.tanh(self.linear1(x))
            x = F.tanh(self.linear2(x))
            x = torch.cat([x, other_dat], axis=1)
            x = F.tanh(self.linear3(x))

            return x.reshape((self.batch_size, self.embedding_dim))  # phil harris way

class Decoder_SPLIT(nn.Module):

    def __init__(self, seq_len, n_features=1, input_dim=64,):
        super(Decoder_SPLIT, self).__init__()
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


import time
class LSTM_AE_SPLIT(nn.Module):

    def __init__(self, num_ifos, num_timesteps, BOTTLENECK):
        super(LSTM_AE_SPLIT, self).__init__()

        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.BOTTLENECK = BOTTLENECK
        self.encoder = Encoder_SPLIT(
            seq_len=num_timesteps, n_features=num_ifos, embedding_dim=BOTTLENECK)
        self.decoder = Decoder_SPLIT(
            seq_len=num_timesteps, n_features=num_ifos, input_dim=BOTTLENECK)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.decoder(x)

        x = x.transpose(1, 2)

        return x


class LSTM_AE_SPLIT_precompute(nn.Module):

    def __init__(self, num_ifos, num_timesteps, BOTTLENECK):
        super(LSTM_AE_SPLIT_precompute, self).__init__()

        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.BOTTLENECK = BOTTLENECK
        self.encoder = Encoder_SPLIT_intermediate(
            seq_len=num_timesteps, n_features=num_ifos, embedding_dim=BOTTLENECK, intermediate_position="return_rnn")
        self.decoder = Decoder_SPLIT(
            seq_len=num_timesteps, n_features=num_ifos, input_dim=BOTTLENECK)

        
    def forward(self, x):
        x, _ = x
        x = x.transpose(1, 2)
        x, batch_size = self.encoder(x)
        return x, batch_size

class LSTM_AE_SPLIT_use_precomputed(nn.Module):

    def __init__(self, num_ifos, num_timesteps, BOTTLENECK, batch_size):
        super(LSTM_AE_SPLIT_use_precomputed, self).__init__()

        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.BOTTLENECK = BOTTLENECK
        self.encoder = Encoder_SPLIT_intermediate(
            seq_len=num_timesteps, n_features=num_ifos, embedding_dim=BOTTLENECK, 
            intermediate_position="return_embedding", batch_size=batch_size)
        self.decoder = Decoder_SPLIT(
            seq_len=num_timesteps, n_features=num_ifos, input_dim=BOTTLENECK)

    def forward(self, x):
        t332 = time.time()
        x = self.encoder(x)
        #print(x[0, 0])
        #print("XX encoding time", time.time()-t332)
        #(a(x))
        #t338 = time.time()
        x = self.decoder(x)
        
        #(a(x))
        x = x.transpose(1, 2)
        #print(x[0, 0, 0])
        #print("XX decoding time", time.time()-t338)
        return x

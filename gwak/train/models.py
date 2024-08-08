import os
import time
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning.pytorch as pl
import scipy.signal as sig

class GwakBaseModelClass(pl.LightningModule):
    pass


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_train_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path, arch=pl_module.model, metric=pl_module.metric
        )

        save_dir = trainer.logger.log_dir or trainer.logger.save_dir
        torch.save(module.model.state_dict(), os.path.join(save_dir, 'model.pt'))


class LargeLinear(GwakBaseModelClass):

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

        x = batch
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

        loss = loss_fn(batch, x)

        self.log(
            'train_loss',
            loss,
            sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x = batch
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


class Autoencoder(GwakBaseModelClass):

    def __init__(
        self,
        num_ifos: int = 2,
        num_timesteps: int = 2048   ,
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
        #print(x.shape)
        #assert 0

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
            sync_dist=True
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


class LearnedSpectrogram1D(nn.Module):
    def __init__(self, L, M, K, N, device, D=None):
        super(LearnedSpectrogram1D, self).__init__()
        '''
        L: sequence length of strain input
        M: number of masks (like count of bins in histogram)
        K: number of frequency features
        
        TODO implement non-linear frequency feature sampling

        D: implement a reduced representation of the masking layer matrix computation
        
        not reduced - use a weight matrix of L x (L*M). 
        input size L gets cast to L*M, then reshaped
        to a LxM masking matrix
        L*L*M weights

        reduced representation - have two weights, 
        one is L x (L*D) and the other is L x (D*M)

        take the inputs and cast to two things:
        THIS IS SOUNDING SO MUCH LIKE A TRANSFORMER
        one to vector L*D, gets reshaped into LxD
        one to vector D*M, gets reshaped into DxM

        then the final LxM matrix computed from
        (LxD)x(DxM) = LxM

        this way, theres L*L*D + L*D*M weights
        = L * (L*D + D*M), so the D condition in
        D << L, M gives big reduction in the 
        number of weights 
        '''
        self.L = L
        self.M = M
        #self.N = L // M # intuitevely the number of windows, at least to scale
        # use it a a predefined parameter - the above way doesn't take into account overlap
        # controls the "rotation rate" of the exponential, and it doesn't work with the wrong value
        self.N = N
        self.K = K
        self.D = D

        # predefine fourier matrix
        self.fourier_matrix = torch.ones(L, K, device=device)
        for l in range(L):
            self.fourier_matrix[l, :] *= l
        for k in range(K):
            self.fourier_matrix[:, k] *= k


        # figure this bit out... maybe the rotations will be too fast?
        # which can be problematic if the mask is sensitive to the entire signal
        # but in principle the fact that it rotates quickly may encourage it to
        # pick a smaller region to be sensitive too..although this could be bad as well
        # since you miss sensitivity to longer signals

        #sigma = torch.exp(-2 * torch.pi * 1j / L)
        sigma = torch.exp(-2 * torch.pi * 1j / self.N  * torch.tensor([1], device=device)) # weird but it doesn't take
                                                                            # complex by itself
        self.fourier_matrix = sigma ** self.fourier_matrix


        # real inputs, those are negative frequency terms
        self.fourier_matrix = self.fourier_matrix[:, :K//2+1] *self.N**0.5 #/ self.N ** 0.5

        # most basic implementation to start
        # use the idea of layers being perturbations to existing set values?
        # no way of algorithmically enforcing this, and in principle 
        # it's fine given that you have the bias value
        # maybe a thought for later
        self.masking_layer = nn.Linear(L, L*M)
        
        self.use_reduced_masking = False
        if D is not None:
            self.use_reduced_masking = True
            self.masking_layer_base0 = nn.Linear(L, L)
            self.masking_layer_base1 = nn.Linear(L, L)
            self.masking_layer_base01 = nn.Linear(L, L)
            self.masking_layer_base11 = nn.Linear(L, L)

            self.masking_layer_RED0 = nn.Linear(L, L*D)
            self.masking_layer_RED1 = nn.Linear(L, D*M)

        parametrize_kernels = False
        if parametrize_kernels:
            self.kernel_center0 = nn.Linear(L, L)
            self.kernel_center1 = nn.Linear(L, L)
            self.kernel_center2 = nn.Linear(L, L)
            


        self.bias_factor_layer = nn.Linear(L, M)
        self.expansion_factor_layer = nn.Linear(L, M)

        # to MxK
        K = K // 2 + 1
        self.frequency_reweighter0 = nn.Linear(L, M*K // 4)
        self.frequency_reweighter1 = nn.Linear(M*K//4, M*K // 4)
        self.frequency_reweighter2 = nn.Linear(M*K//4, M*K // 4)
        self.frequency_reweighter3 = nn.Linear(M*K//4, M*K)

    
        self.softmax = nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()

    def forward(self, x, return_attention=False, use_FRW=True):
        '''
        input x is of shape (N_batch, L)
        '''     
        NB, L_ = x.shape
        assert L_ == self.L
        x = x - torch.mean(x, axis=1)[:, None]
        # compute "attn" weights
        if self.use_reduced_masking:
            #print(100, "x"); check(x)
            deeper = True
            if deeper:
                A0 = self.ReLU(self.masking_layer_base0(x))
                A1 = self.ReLU(self.masking_layer_base1(x))
                even_deeper = True
                if even_deeper:
                    A0 = self.ReLU(self.masking_layer_base01(A0))
                    A1 = self.ReLU(self.masking_layer_base11(A1))

            else:
                A0 = x
                A1 = x
            A0 = self.masking_layer_RED0(A0) # (NB, L*D)
            A0 = torch.reshape(A0, (NB, self.L, self.D)) # (NB, L, D)

            
            A1 = self.masking_layer_RED1(A1) # (NB, D*M)
            A1 = torch.reshape(A1, (NB, self.D, self.M)) # (NB, D, M)
            #print(104, "A0, A1"); check(A0); check(A1)
            A = torch.matmul(A0, A1) # (NB, L, M)
            #print(105, "A"); check(A)
        else:
            A = self.masking_layer(x) # (NB, L*M)
            A = torch.reshape(A, (NB, self.L, self.M)) # (NB, L, M)

        assert A.shape[1] == self.L and A.shape[2] == self.M

        #A = self.softmax(A)
        A = self.ReLU(A)
        #A = torch.exp(A)

        # perform a softmax across the attention matrix,
        # the same way that it's done in a transformer
        if 1:
            # compute the bias and expansion factors
            bias_factors = self.bias_factor_layer(x) # (NB, M)
            # note the use of self.N to set a dimensionless scale for the bias_factor
            bias_factors = torch.exp(-2 * torch.pi * 1j * self.N * bias_factors) # (NB, M)
            expansion_factors = self.expansion_factor_layer(x) # (NB, M)
            # same idea with the "dimensionless scale", although here it's a rescaling but already dimless
            expansion_factors = expansion_factors * self.L / self.N

        if 1:
            FRW = self.ReLU(self.frequency_reweighter0(x))
            FRW = self.ReLU(self.frequency_reweighter1(FRW))
            FRW = self.ReLU(self.frequency_reweighter2(FRW))
            FRW = self.frequency_reweighter3(FRW)
            FRW = torch.reshape(FRW, (NB, self.M, self.K//2 + 1))
            FRW = self.softmax(FRW)


        # reshape and copy over axis for "attn" multiplication
        x = x[:, :, None] # (NB, L, 1)
        x = x.repeat(1, 1, self.M) # (NB, L, M)
        # apply attention/mask
        x = torch.mul(x, A) #(NB, L, M) (NB, L, M)
        # apply exponential bias factors (exponential multiplcation = addition, hence bias)
        x = torch.mul(x, bias_factors[:, None, :]) # (NB, L, M) 
        # not sure how to implement the multiplicative factors
        # TODO figure out the math
        # apply the fourier matrix
        x = torch.swapaxes(x, 1, 2) # (NB, L, M) -> (NB, M, L)
        x = torch.matmul(x, self.fourier_matrix) # (NB, M, L) x (L, K) = (NB, M, K)
        spectrogram = torch.abs(x)#*self.N**2# ** 2 # back to real, same way spectrogram = | STFT | ** 2
        if use_FRW:
            spectrogram = torch.mul(spectrogram, FRW)
        if return_attention:
            return spectrogram, A
        return spectrogram 

class BaseSpectrogram1D(nn.Module):
    def __init__(self, L, M, K, N, device):
        super(BaseSpectrogram1D, self).__init__()
        
        #self.window = torch.ones(N, device=device) / N
        self.window = sig.windows.tukey(N, alpha=0.25)
        self.window /= self.window.sum()
        self.window = torch.from_numpy(self.window).to(device).float()
        self.window_stride = (L-N) // (M - 1)
        assert self.window_stride * (M-1) + N == L
        print(180, self.window_stride)
        self.L = L
        self.M = M
        self.K = K
        self.N = N
                
        # predefine fourier matrix
        self.fourier_matrix = torch.ones(N, K, device=device)
        for n in range(N):
            self.fourier_matrix[n, :] *= n
        for k in range(K):
            self.fourier_matrix[:, k] *= k
            
        sigma = torch.exp(-2 * torch.pi * 1j / self.N * torch.tensor([1], device=device))
        self.fourier_matrix = sigma ** self.fourier_matrix

        self.fourier_matrix = self.fourier_matrix[:, :K//2+1] * self.N**0.5 #/ self.N ** 0.5
                
        #print("nperseg", N, "noverlap", self.window_stride, "nfft", K)
    def forward(self, x):
        # get the hankelization
        x = x + 0j
        x = x - torch.mean(x, axis=1)[:, None]
        x = x.unfold(1, self.N, self.window_stride)
        
        # apply the window
        x = torch.mul(x, self.window[None, None, :])
        #return x
        
        x = torch.matmul(x, self.fourier_matrix)
        

        return torch.abs(x)

    
class LearnedSpectrogram2D(nn.Module):
    def __init__(self, L, M, K, N, device, D=None):
        super(LearnedSpectrogram2D, self).__init__()
        '''
        L: sequence length of strain input
        M: number of masks (like count of bins in histogram)
        K: number of frequency features
        
        TODO implement non-linear frequency feature sampling

        D: implement a reduced representation of the masking layer matrix computation
        
        not reduced - use a weight matrix of L x (L*M). 
        input size L gets cast to L*M, then reshaped
        to a LxM masking matrix
        L*L*M weights

        reduced representation - have two weights, 
        one is L x (L*D) and the other is L x (D*M)

        take the inputs and cast to two things:
        THIS IS SOUNDING SO MUCH LIKE A TRANSFORMER
        one to vector L*D, gets reshaped into LxD
        one to vector D*M, gets reshaped into DxM

        then the final LxM matrix computed from
        (LxD)x(DxM) = LxM

        this way, theres L*L*D + L*D*M weights
        = L * (L*D + D*M), so the D condition in
        D << L, M gives big reduction in the 
        number of weights 
        '''
        self.L = L
        self.M = M
        #self.N = L // M # intuitevely the number of windows, at least to scale
        # use it a a predefined parameter - the above way doesn't take into account overlap
        # controls the "rotation rate" of the exponential, and it doesn't work with the wrong value
        self.N = N
        self.K = K
        self.D = D

        # predefine fourier matrix
        self.fourier_matrix = torch.ones(L, K, device=device)
        for l in range(L):
            self.fourier_matrix[l, :] *= l
        for k in range(K):
            self.fourier_matrix[:, k] *= k


        # figure this bit out... maybe the rotations will be too fast?
        # which can be problematic if the mask is sensitive to the entire signal
        # but in principle the fact that it rotates quickly may encourage it to
        # pick a smaller region to be sensitive too..although this could be bad as well
        # since you miss sensitivity to longer signals

        #sigma = torch.exp(-2 * torch.pi * 1j / L)
        sigma = torch.exp(-2 * torch.pi * 1j / self.N  * torch.tensor([1], device=device)) # weird but it doesn't take
                                                                            # complex by itself
        self.fourier_matrix = sigma ** self.fourier_matrix


        # real inputs, those are negative frequency terms
        self.fourier_matrix = self.fourier_matrix[:, :K//2+1] *self.N**0.5 #/ self.N ** 0.5

        # most basic implementation to start
        # use the idea of layers being perturbations to existing set values?
        # no way of algorithmically enforcing this, and in principle 
        # it's fine given that you have the bias value
        # maybe a thought for later
        self.masking_layer = nn.Linear(L, L*M)
        
        self.use_reduced_masking = False
        if D is not None:
            self.use_reduced_masking = True
            self.masking_layer_base0 = nn.Linear(L, L)
            self.masking_layer_base1 = nn.Linear(L, L)
            self.masking_layer_base01 = nn.Linear(L, L)
            self.masking_layer_base11 = nn.Linear(L, L)

            self.masking_layer_RED0 = nn.Linear(L, L*D)
            self.masking_layer_RED1 = nn.Linear(L, D*M)




        parametrize_kernels = False
        if parametrize_kernels:
            self.kernel_center0 = nn.Linear(L, L)
            self.kernel_center1 = nn.Linear(L, L)
            self.kernel_center2 = nn.Linear(L, L)
            


        self.bias_factor_layerA = nn.Linear(L, M)
        self.bias_factor_layerB = nn.Linear(L, M)
        self.expansion_factor_layer = nn.Linear(L, M)

        # to MxK
        K = K // 2 + 1
        self.frequency_reweighter0A = nn.Linear(L, M*K // 4)
        self.frequency_reweighter1A = nn.Linear(M*K//4, M*K // 4)
        self.frequency_reweighter2A = nn.Linear(M*K//4, M*K // 4)
        self.frequency_reweighter3A = nn.Linear(M*K//4, M*K)

        self.frequency_reweighter0B = nn.Linear(L, M*K // 4)
        self.frequency_reweighter1B = nn.Linear(M*K//4, M*K // 4)
        self.frequency_reweighter2B = nn.Linear(M*K//4, M*K // 4)
        self.frequency_reweighter3B = nn.Linear(M*K//4, M*K)

    
        self.softmax = nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()

    def forward(self, x, return_attention=False):
        '''
        input x is of shape (N_batch, 2, L)
        '''     
        NB, _, L_ = x.shape
        assert L_ == self.L
        xA = x[:, 0, :]
        xB = x[:, 1, :]

        xA = xA - torch.mean(xA, axis=1)[:, None]
        xB = xB - torch.mean(xB, axis=1)[:, None]
        # compute "attn" weights
        if self.use_reduced_masking:
            #print(100, "x"); check(x)
            deeper = True
            if deeper:
                T0A = self.ReLU(self.masking_layer_base0(xA))
                T1A = self.ReLU(self.masking_layer_base1(xA))
                even_deeper = True
                if even_deeper:
                    T0A = self.ReLU(self.masking_layer_base01(T0A))
                    T1A = self.ReLU(self.masking_layer_base11(T1A))

            else:
                T0A = xA
                T1A = xA
            
            T0A = self.masking_layer_RED0(T0A) # (NB, L*D)
            T0A = torch.reshape(T0A, (NB, self.L, self.D)) # (NB, L, D)

            
            T1A = self.masking_layer_RED1(T1A) # (NB, D*M)
            T1A = torch.reshape(T1A, (NB, self.D, self.M)) # (NB, D, M)
            #print(104, "A0, A1"); check(A0); check(A1)
            TA = torch.matmul(T0A, T1A) # (NB, L, M)
            #print(105, "A"); check(A)
        else:
            TA = self.masking_layer(xA) # (NB, L*M)
            TA = torch.reshape(TA, (NB, self.L, self.M)) # (NB, L, M)

        if self.use_reduced_masking:
            #print(100, "x"); check(x)
            deeper = True
            if deeper:
                T0B = self.ReLU(self.masking_layer_base0(xB))
                T1B = self.ReLU(self.masking_layer_base1(xB))
                even_deeper = True
                if even_deeper:
                    T0B = self.ReLU(self.masking_layer_base01(T0B))
                    T1B = self.ReLU(self.masking_layer_base11(T1B))

            else:
                T0B = xB
                T1B = xB
            
            T0B = self.masking_layer_RED0(T0B) # (NB, L*D)
            T0B = torch.reshape(T0B, (NB, self.L, self.D)) # (NB, L, D)

            
            T1B = self.masking_layer_RED1(T1B) # (NB, D*M)
            T1B = torch.reshape(T1B, (NB, self.D, self.M)) # (NB, D, M)
            #print(104, "A0, A1"); check(A0); check(A1)
            TB = torch.matmul(T0B, T1B) # (NB, L, M)
            #print(105, "A"); check(A)
        else:
            TB = self.masking_layer(xB) # (NB, L*M)
            TB = torch.reshape(TB, (NB, self.L, self.M)) # (NB, L, M)

        #assert A.shape[1] == self.L and A.shape[2] == self.M

        #A = self.softmax(A)
        TA = self.ReLU(TA)
        TB = self.ReLU(TB)

        # perform a softmax across the attention matrix,
        # the same way that it's done in a transformer
        if 1:
            # compute the bias and expansion factors
            bias_factorsA = self.bias_factor_layerA(xA) # (NB, M)
            # note the use of self.N to set a dimensionless scale for the bias_factor
            bias_factorsA = torch.exp(-2 * torch.pi * 1j * self.N * bias_factorsA) # (NB, M)

            bias_factorsB = self.bias_factor_layerB(xB) # (NB, M)
            bias_factorsB = torch.exp(-2 * torch.pi * 1j * self.N * bias_factorsB) # (NB, M)



            if 0:
                expansion_factors = self.expansion_factor_layer(x) # (NB, M)
                # same idea with the "dimensionless scale", although here it's a rescaling but already dimless
                expansion_factors = expansion_factors * self.L / self.N

        if 1:
            FRWA = self.ReLU(self.frequency_reweighter0A(xA))
            FRWA = self.ReLU(self.frequency_reweighter1A(FRWA))
            FRWA = self.ReLU(self.frequency_reweighter2A(FRWA))
            FRWA = self.frequency_reweighter3A(FRWA)
            FRWA = torch.reshape(FRWA, (NB, self.M, self.K//2 + 1))
            FRWA = self.softmax(FRWA)

            FRWB = self.ReLU(self.frequency_reweighter0B(xB))
            FRWB = self.ReLU(self.frequency_reweighter1B(FRWB))
            FRWB = self.ReLU(self.frequency_reweighter2B(FRWB))
            FRWB = self.frequency_reweighter3B(FRWB)
            FRWB = torch.reshape(FRWB, (NB, self.M, self.K//2 + 1))
            FRWB = self.softmax(FRWB)


        # reshape and copy over axis for "attn" multiplication
        xA = xA[:, :, None] # (NB, L, 1)
        xA = xA.repeat(1, 1, self.M) # (NB, L, M)
        # apply attention/mask

        # SPICY: SWAPPING THE MASKS
        xA = torch.mul(xA, TB) #(NB, L, M) (NB, L, M)
        # apply exponential bias factors (exponential multiplcation = addition, hence bias)
        xA = torch.mul(xA, bias_factorsA[:, None, :]) # (NB, L, M) 
        # not sure how to implement the multiplicative factors
        # TODO figure out the math
        # apply the fourier matrix
        xA = torch.swapaxes(xA, 1, 2) # (NB, L, M) -> (NB, M, L)
        xA = torch.matmul(xA, self.fourier_matrix) # (NB, M, L) x (L, K) = (NB, M, K)
        spectrogramA = torch.abs(xA)#*self.N**2# ** 2 # back to real, same way spectrogram = | STFT | ** 2

        use_freq_mat = True
        if use_freq_mat:
            spectrogramA = torch.mul(spectrogramA, FRWB)

        xB = xB[:, :, None] # (NB, L, 1)
        xB = xB.repeat(1, 1, self.M) # (NB, L, M)

        xB = torch.mul(xB, TA) #(NB, L, M) (NB, L, M)
        # apply exponential bias factors (exponential multiplcation = addition, hence bias)
        xB = torch.mul(xB, bias_factorsB[:, None, :]) # (NB, L, M) 
        # not sure how to implement the multiplicative factors
        # TODO figure out the math
        # apply the fourier matrix
        xB = torch.swapaxes(xB, 1, 2) # (NB, L, M) -> (NB, M, L)
        xB = torch.matmul(xB, self.fourier_matrix) # (NB, M, L) x (L, K) = (NB, M, K)
        spectrogramB = torch.abs(xB)#*self.N**2# ** 2 # back to real, same way spectrogram = | STFT | ** 2

        if use_freq_mat:

            spectrogramB = torch.mul(spectrogramB, FRWA)

        spectrogram = torch.cat([spectrogramA[:, None, :, :], spectrogramB[:, None, :, :]], axis=1)
        T = torch.cat([TA[:, None], TB[:, None]], axis=1)
        if return_attention:
            return spectrogram, T
        return spectrogram 

class BaseSpectrogram2D(nn.Module):
    def __init__(self, L, M, K, N, device):
        super(BaseSpectrogram2D, self).__init__()
        
        #self.window = torch.ones(N, device=device) / N
        self.window = sig.windows.tukey(N, alpha=0.25)
        self.window /= self.window.sum()
        self.window = torch.from_numpy(self.window).to(device).float()
        self.window_stride = (L-N) // (M - 1)
        assert self.window_stride * (M-1) + N == L
        print(180, self.window_stride)
        self.L = L
        self.M = M
        self.K = K
        self.N = N
                
        # predefine fourier matrix
        self.fourier_matrix = torch.ones(N, K, device=device)
        for n in range(N):
            self.fourier_matrix[n, :] *= n
        for k in range(K):
            self.fourier_matrix[:, k] *= k
            
        sigma = torch.exp(-2 * torch.pi * 1j / self.N * torch.tensor([1], device=device))
        self.fourier_matrix = sigma ** self.fourier_matrix

        self.fourier_matrix = self.fourier_matrix[:, :K//2+1] * self.N**0.5 #/ self.N ** 0.5
                
        #print("nperseg", N, "noverlap", self.window_stride, "nfft", K)
    def forward(self, x):
        # get the hankelization
        x = x + 0j
        xA = x[:, 0, :]
        xB = x[:, 1, :]

        xA = xA - torch.mean(xA, axis=1)[:, None]
        xA = xA.unfold(1, self.N, self.window_stride)
        
        # apply the window
        xA = torch.mul(xA, self.window[None, None, :])
        #return x
        
        xA = torch.matmul(xA, self.fourier_matrix)

        ##b:
        xB = xB - torch.mean(xB, axis=1)[:, None]
        xB = xB.unfold(1, self.N, self.window_stride)
        
        # apply the window
        xB = torch.mul(xB, self.window[None, None, :])
        #return x
        
        xB = torch.matmul(xB, self.fourier_matrix)
        
        xA = torch.abs(xA)
        xB = torch.abs(xB)
        return torch.cat([xA[:, None, :, :], xB[:, None, :, :]], axis=1)

class EncoderTransformer(nn.Module):
    def __init__(self, num_timesteps, num_features, latent_dim, nhead, dim_feedforward, device):
        super(EncoderTransformer, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.latent_dim = latent_dim

        #dim_feedforward = 128
        #nhead=4
        print("nhead, latent, num_features", nhead, latent_dim, num_features)
        self.transformer1 = nn.TransformerEncoderLayer(d_model=num_features*2,  
                                                       nhead=nhead, 
                                                       dim_feedforward=dim_feedforward,
                                                       batch_first=True).to(device)
        self.layer1 = nn.Linear(num_features*2, num_features)
        self.transformer2 = nn.TransformerEncoderLayer(d_model=num_features,  nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True).to(device)
        self.layer2 = nn.Linear(num_features, num_features//2)
        self.transformer3 = nn.TransformerEncoderLayer(d_model=num_features//2,  nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True).to(device)
        
        self.layer3 = nn.Linear(num_timesteps*(num_features//2), latent_dim * 4)
        self.layer4 = nn.Linear(latent_dim*4, latent_dim*4)
        self.layer5 = nn.Linear(latent_dim*4, latent_dim)

    def forward(self, x):
        # (NB, N_IFO, N_TIMESTEPS, N_FEATURES) -> (NB, latent_dim)
        n_batches, n_ifos, num_timesteps, num_features = x.shape
        assert n_ifos==2
        x = x.swapaxes(1, 2)
        x = x.reshape((n_batches, num_timesteps, num_features*2))

        x = self.transformer1(x)
        x = F.relu(self.layer1(x))

        x = self.transformer2(x)
        x = F.relu(self.layer2(x))
        
        x = self.transformer3(x)
        x = x.reshape( (n_batches, self.num_timesteps * (self.num_features//2)))

        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)

        return x

class DecoderTransformer(nn.Module):
    def __init__(self, num_timesteps, num_features, latent_dim, nhead, dim_feedforward, device):
        super(DecoderTransformer, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.latent_dim = latent_dim

        
        self.transformer1 = nn.TransformerEncoderLayer(d_model=num_features*2,  nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True).to(device)
        self.layer1 = nn.Linear(num_features, num_features*2)
        self.transformer2 = nn.TransformerEncoderLayer(d_model=num_features,  nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True).to(device)
        self.layer2 = nn.Linear(num_features//2, num_features)
        self.transformer3 = nn.TransformerEncoderLayer(d_model=num_features//2,  nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True).to(device)
        
        self.layer3 = nn.Linear(latent_dim * 4, num_timesteps*(num_features//2))
        self.layer4 = nn.Linear(latent_dim*4, latent_dim*4)
        self.layer5 = nn.Linear(latent_dim, latent_dim*4)

    def forward(self, x):
        # (NB, latent_dim) -> (NB, N_IFO, N_TIMESTEPS, N_FEATURES)
        n_batches,  latent_dim = x.shape
        #assert n_ifos==2
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer3(x))
        x = x.reshape( (n_batches, self.num_timesteps,  (self.num_features//2)))
        x = self.transformer3(x)
        x = F.relu(self.layer2(x))
        x = self.transformer2(x)
        x = F.relu(self.layer1(x))
        x = self.transformer1(x)
        x = x.reshape((n_batches, self.num_timesteps, 2, self.num_features))

        x = x.swapaxes(1, 2)
        return x
    
class VICRegLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.relu = torch.nn.ReLU()

    def forward(self, z_a, z_b):
        # https://arxiv.org/pdf/2105.04906.pdf#page=12&zoom=100,110,840
        N, D = z_a.shape
        sim_loss = self.mse_loss(z_a, z_b)
        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(self.relu(1 - std_z_a)) + torch.mean(self.relu(1 - std_z_b))
        # covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)
        cov_loss = (self.off_diagonal(cov_z_a)**2).sum() / D + (self.off_diagonal(cov_z_b)**2).sum() / D
        # loss
        # weights?
        lambda_weight, mu_weight, nu_weight = 1, 1, 1
        loss = lambda_weight * sim_loss + mu_weight * std_loss + nu_weight * cov_loss
        return loss

    def off_diagonal(self, x):
        N, D = x.shape
        mask = 1 - torch.eye(n=D, device=x.device)
        return x * mask


class gwak2(GwakBaseModelClass):

    def __init__(self, 
                num_ifos:int=2, 
                num_timesteps:int=2048,
                latent_space_dim:int=16,

                specgram_params:dict={"M":65, 
                                      "K":62, # gets //2 + 1, which gives 32, which behaves nicely with transformer
                                      "N":64},

                transformer_params:dict={"nhead": 4,
                                        "dim_feedforward":1024},

                spectrogram_type:str="base",
                device_str:str="cuda:0" # some default to initialize model weights
                                    # moved to gpu later 
                                    # TODO: implement better, remove from specgram class defn.
                ):

        super().__init__()
        
        # initialize the spectrogram
        # add possibility for num_ifos=1 to give 1D specgram?
        # don't know if we will use it

        self.num_ifos = num_ifos
        self.num_timesteps = num_timesteps
        self.latent_space_dim = latent_space_dim
        self.specgram_params = specgram_params
        self.transformer_params = transformer_params
        self.device_str = device_str
        
        if spectrogram_type == "base":
            assert self.num_ifos == 2
            self.spectrogram = BaseSpectrogram2D(self.num_timesteps,
                                                self.specgram_params["M"],
                                                self.specgram_params["K"],
                                                self.specgram_params["N"],
                                                torch.device(self.device_str))
        else:
            assert self.num_ifos == 2
            self.spectrogram = LearnedSpectrogram2D(self.num_timesteps,
                                                self.specgram_params["M"],
                                                self.specgram_params["K"],
                                                self.specgram_params["N"],
                                                torch.device(self.device_str))
        
        self.encoder = EncoderTransformer(num_timesteps = self.specgram_params["M"],
                                        num_features = self.specgram_params["K"] // 2 + 1, 
                                        latent_dim = self.latent_space_dim,
                                        nhead = self.transformer_params["nhead"],
                                        dim_feedforward = self.transformer_params["dim_feedforward"],
                                        device = torch.device(self.device_str))
        # don't know if this will be used... maybe remove
        # depends on the training objective
        self.decoder = DecoderTransformer(num_timesteps = self.specgram_params["M"],
                                        num_features = self.specgram_params["K"] // 2 + 1, 
                                        latent_dim = self.latent_space_dim,
                                        nhead = self.transformer_params["nhead"],
                                        dim_feedforward = self.transformer_params["dim_feedforward"],
                                        device = torch.device(self.device_str))               
        
        # loss functions
        self.vicreg_loss = VICRegLoss()
        self.l1_loss = nn.L1Loss()
        

    def training_step(self, batch, batch_idx):
        x = batch
        x = self.spectrogram(x)
        latent = self.encoder(x)
        recon = self.decoder(latent)

        loss = self.l1_loss(x, recon)


        self.log(
            'train_loss',
            loss,
            sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

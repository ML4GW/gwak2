class SignalDataloaderAugmentation(GwakBaseDataloader):
    def __init__(
        self, 
        prior: data.BasePrior, 
        waveform: torch.nn.Module, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.waveform = waveform
        self.prior = prior
        
        # Projection parameters
        self.ra_prior =  Uniform(0, 2*torch.pi)
        self.dec_prior = Cosine(-np.pi/2, torch.pi/2)
        self.phic_prior = Uniform(0, 2 * torch.pi)

        self.sky_location_augmentation = True
        self.distance_augmentation = False
        self.tc_augmentation = False


    def generate_waveforms_augmented(self, batch_size):

        # get detector orientations
        ifos = ['H1', 'L1']
        tensors, vertices = get_ifo_geometry(*ifos)

        # sample from prior and generate waveforms
        parameters = self.prior.sample(batch_size) # dict[str, torch.tensor]

        #FIRST "AUGMENTATION", i.e. no transformation
        
        ra = self.ra_prior.sample((batch_size,))
        dec = self.dec_prior.sample((batch_size,))
        
        cross, plus = self.waveform(**parameters)
        cross, plus = torch.fft.irfft(cross), torch.fft.irfft(plus)
        # Normalization
        cross *= self.sample_rate
        plus *= self.sample_rate

        # roll the waveforms to join
        # the coalescence and ringdown
        cross = torch.roll(cross, -self.ringdown_size, dims=-1)
        plus = torch.roll(plus, -self.ringdown_size, dims=-1)

        # compute detector responses
        responses_aug0 = compute_observed_strain(
            # parameters['dec'],
            dec,
            parameters['phic'], # psi
            # parameters['ra'],
            ra,
            tensors,
            vertices,
            self.sample_rate,
            cross=cross.float(),
            plus=plus.float()
        ).to('cuda')

        # SECOND AUGMENTATION
        if self.sky_location_augmentation: #reroll
            ra = self.ra_prior.sample((batch_size,))
            dec = self.dec_prior.sample((batch_size,))
        
        parameters_regenerated = self.prior.sample(batch_size)
        if self.distance_augmentation:
            parameters['distance'] = parameters_regenerated['distance']

        if self.tc_augmentation:
            # prior sets everything to zero, so implement this later
            None 
        
        cross, plus = self.waveform(**parameters)
        cross, plus = torch.fft.irfft(cross), torch.fft.irfft(plus)
        # Normalization
        cross *= self.sample_rate
        plus *= self.sample_rate

        # roll the waveforms to join
        # the coalescence and ringdown
        cross = torch.roll(cross, -self.ringdown_size, dims=-1)
        plus = torch.roll(plus, -self.ringdown_size, dims=-1)

        # compute detector responses
        responses_aug1 = compute_observed_strain(
            # parameters['dec'],
            dec,
            parameters['phic'], # psi
            # parameters['ra'],
            ra,
            tensors,
            vertices,
            self.sample_rate,
            cross=cross.float(),
            plus=plus.float()
        ).to('cuda')


        return torch.stack([responses_aug0, responses_aug1])

    def inject(self, batch, waveforms):
        #print(553, waveforms.shape)
        aug_0, aug_1 = waveforms
        
        aug_0_injected = self.inject_individual(batch, aug_0)
        aug_1_injected = self.inject_individual(batch, aug_1)
        return torch.stack([aug_0_injected, aug_1_injected])

    def inject_individual(self, batch, waveforms):
        #for augmentation in batch:

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

        # Waveform padding
        inj_len = waveforms.shape[-1]
        window_len = splits[1]
        half = int((window_len - inj_len)/2)

        first_half, second_half = half, window_len - half - inj_len
        
        waveforms = F.pad(
            input=waveforms, 
            pad=(first_half, second_half), 
            mode='constant', 
            value=0
        )
        
        injected = batch + waveforms * 100

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
            waveforms = self.generate_waveforms_augmented(batch.shape[0])
            # inject waveforms; maybe also whiten data preprocess etc..
            
            batch = self.inject(batch, waveforms)
            
            if self.trainer.training and (self.data_saving_file is not None):
                
                # Set a warning that when the global_step exceed 1e6, 
                # the data will have duplications. 
                # Replace this with a data saving function. 
                bk_step = f"Training/Step_{self.trainer.global_step:06d}_BK"
                inj_step = f"Training/Step_{self.trainer.global_step:06d}_INJ"
                
                self.data_group.create_dataset(bk_step, data = batch.cpu())
                self.data_group.create_dataset(inj_step, data = waveforms.cpu())
                
            if self.trainer.validating and (self.data_saving_file is not None):

                bk_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_BK"
                inj_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_INJ"
                
                self.data_group.create_dataset(bk_step, data = batch.cpu())
                self.data_group.create_dataset(inj_step, data = waveforms.cpu())          
                
            return batch

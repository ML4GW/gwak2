class Prior():

    def __init__(self, **kwargs):

        self.params = dict()
        for k, v in kwargs.items():
            print(f'Setting for key {k}, value {v}')
            self.params[k] = v

    def sample(self, batch_size):

        self.sampled_params = dict()

        for k in self.params.keys():
            self.sampled_params[k] = self.params[k].sample((batch_size,))

        return self.sampled_params
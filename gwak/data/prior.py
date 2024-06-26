class Prior():

    def __init__(self, **kwargs):

        self.params = dict()
        for k, v in kwargs.items():
            self.params[k] = v

    def sample(self, batch_size):

        for k in self.params.keys():
            self.sampled_params[k] = self.params.sample((batch_size,))

        return self.sampled_params
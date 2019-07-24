import pprint


class Config():
    """Config class"""
    def __init__(self, **kwargs):

        # Path
        self.data_path = '../data/fcsn_tvsum.h5'
        self.save_dir = 'save_dir'
        self.score_dir = 'score_dir'
        self.log_dir = 'log_dir'

        # Model
        self.mode = 'train'
        self.gpu = False
        self.n_epochs = 50
        self.n_class = 2
        self.lr = 1e-3
        self.momentum = 0.9
        self.batch_size = 5

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        config_str = 'Configurations\n' + pprint.pformat(self.__dict__)
        return config_str


if __name__ == '__main__':
    config = Config()
    print(config)

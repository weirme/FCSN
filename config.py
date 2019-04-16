# -*- coding: utf-8 -*-

import pprint


class Config():
    """Config class"""
    def __init__(self, **kwargs):

        super(Config, self).__init__()

        # Path
        self.data_path = '/Users/sameal/Downloads/Datasets/TVSum/fcsn_dataset.h5'
        self.save_dir = 'save_dir'
        self.score_dir = 'score_dir'
        self.log_dir = 'log_dir'

        # Model
        self.mode = 'train'
        self.n_epochs = 50
        self.n_class = 2
        self.lr = 1e-3
        self.batch_size = 5

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        config_str = 'Configurations\n' + pprint.pformat(self.__dict__)
        return config_str


if __name__ == '__main__':
    config = Config()
    print(config)

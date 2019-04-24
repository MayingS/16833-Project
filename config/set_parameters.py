class Params:
    def __init__(self):
        self.globalparam = {
            'min_obs_likelihood': 0.004,
            'workers': 4,
            'init_with_true_state': False,
            'use_proposer': False,
            'propose_ratio': 0,
            'proposer_keep_ratio': 0.15
        }
        self.train = {
            'train_individually': True,
            'train_e2e': True,
            'batch_size': 32,
            'epochs': 10000,
            'seq_length': 20,
            'learning_rate': 0.0003,
            'dropout_keep_ratio': 0.3,
            'particle_num': 100,
            'train_motion_model_mode':0
        }
        self.test = {
            'seq_length': 50,
            'particle_num': 1000
        }

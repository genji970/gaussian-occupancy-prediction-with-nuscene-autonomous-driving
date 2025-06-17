from config import config

lifter_cfg = dict(
    type=config['lifter_cfg_type'],
    in_channels=config['lifter_in_channels'],
    num_levels=config['lifter_num_levels'],
    num_gaussians=config['lifter_num_gaussians'],
    hidden_dim=config['lifter_hidden_dim']
)

from os.path import join

class ConfigArgs:
    loss_type = 'wgan-gp'
    log_dir = 'logs'
    sample_dir = join(log_dir, 'samples')
    ckpt_dir = join(log_dir, 'ckpt')
    data_dir = '/home/yangyangii/data/CelebaHQ256'
    retrain = False
    load_model = None # 
    batch_size = 16
    lr = 0.001
    zdim = 256 # == first # of conv maps
    max_step = 1000000
    log_step = 1000
    save_step = 2000
    scale_update_schedule = [
        [80000],
        [20000, 20000, 20000, 20000], # 4x4 -> 8x8
        [30000, 30000, 30000, 30000], # 8x8 -> 16x16
        [40000, 40000, 40000, 40000], # 16x16 -> 32x32
        [50000, 50000, 50000, 50000], # 32x32 -> 64x64
        [50000, 50000, 50000, 50000], # 64x64 -> 128x128
    ]
    scale_update_alpha = [
        [1.0],
        [0.25, 0.5, 0.75, 1.0],
        [0.25, 0.5, 0.75, 1.0],
        [0.25, 0.5, 0.75, 1.0],
        [0.25, 0.5, 0.75, 1.0],
        [0.25, 0.5, 0.75, 1.0],
    ]

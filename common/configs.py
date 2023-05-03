from yacs.config import CfgNode as CN

__C = CN()
__C.epoch = 50
# model configs
__C.model = CN()
__C.model.path = './weights'
__C.model.path = './best'
# dataset configs
__C.dataset = CN()
__C.dataset.batch_size = 128
__C.dataset.path = './img_align_celeba'
__C.dataset.num_workers = 8
# optimizer configs
__C.optimizer = CN()
__C.optimizer.name = "AdamW"
__C.optimizer.base_lr = 0.0005
__C.optimizer.weight_decay = 0.00005
__C.optimizer.betas = (0.9,0.999)
# scheduler configs
__C.scheduler = CN()
__C.scheduler.name = "CosineAnnealingwithWarmRestarts"
__C.scheduler.T_0 = 10
__C.scheduler.T_mult = 2
# logger configs
__C.logger = CN()
__C.logger.path = 'logs'
__C.logger.name = 'train_log.log'

def get_cfg_defaults():
    return __C.clone

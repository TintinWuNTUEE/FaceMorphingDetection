import torch.optim as optim

def build_optimizer(cfg, model):
  lr  = cfg.optimizer.base_lr
  decay = cfg.optimizer.weight_decay
  betas = cfg.optimizer.betas
  optimizer = optim.AdamW(model.parameters(),eps=1e-6,
                                           lr=lr,weight_decay=decay,
                                           betas=betas)
  return optimizer


def build_scheduler(cfg,optimizer):
  T_0 = cfg.scheduler.T_0
  T_MULT = cfg.scheduler.T_mult
  scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = T_0, T_mult=T_MULT)
  return scheduler
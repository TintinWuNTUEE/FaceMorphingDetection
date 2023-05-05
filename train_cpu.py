import torch
from torch import nn
from data import get_loader
from common.utils import get_device
from common.logger import get_logger
from common.optimizer import build_optimizer,build_scheduler
from common.checkpoint import load_checkpoint,save_checkpoint
from common.configs import get_cfg_defaults
from model.model import get_model
device = 'cpu'
def train(logger,cfg,log_interval=200,val_interval = 5,model_name='swin'):
    #training
    criterion = nn.BCEWithLogitsLoss()
    acc = 0
    iteration = 0
    best_acc = 0
    max_epoch =cfg.train.epochs
    patience = 5
    last_loss = 1000
    trigger_times = 0
    max_k_folds = cfg.train.k_folds #5
    for kf in range(0,max_k_folds):
        #get model(model.py)
        logger.info('train '+model_name+' => Loading network architecture...')

        #build optimizer
        logger.info('=> Loading optimizer...')
        optimizer = build_optimizer(cfg,model)

        #build scheduler
        logger.info('=> Loading scheduler...')
        scheduler = build_scheduler(cfg,optimizer)
        scaler = torch.cuda.amp.GradScaler() 
        #get dataset(data.py)
        train_loader,val_loader,test_loader = get_loader(
        cfg,cfg.dataset.path,
        batch_size=cfg.dataset.batch_size,num_workers=cfg.dataset.num_workers,kf=kf)
        #load checkpoint
        model = get_model()
        model=model.to(device)
        model,optimizer,scheduler,scaler,epoch = load_checkpoint(model,optimizer,
        scheduler,scaler,cfg.model.path,
        logger,kf)
        for ep in range(epoch,max_epoch):
            model.train() 
            for batch_idx, (data, label) in enumerate(train_loader):
                data,label= data.to(device), label.to(device,dtype=torch.float)
                optimizer.zero_grad()
                output = model(data).squeeze()
                loss = criterion(output,label)
                loss.backward()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
                scheduler.step(ep + batch_idx / len(train_loader))
                if iteration % log_interval == 0:
                    logger.info('Train '+model_name+' Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f},lr:{:.4f}'.format(
                    ep, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),scheduler.get_last_lr()[0]))
                iteration += 1
                del data,label
                torch.cuda.empty_cache()
            if ep%val_interval == 0:
                test_loss,acc= validation(model,val_loader,logger)
            if test_loss > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    logger.info('Early stopping!\n')
                    return
                last_loss = test_loss
            else:
                trigger_times = 0
                save_checkpoint(cfg.model.path,model,optimizer,ep,scheduler,scaler,kf)
                logger.info(f'saving model with loss: {test_loss}')
                last_loss = test_loss
            if acc>best_acc:
                best_acc = acc
                logger.info(f'saving model with acc: {acc}')
                save_checkpoint(cfg.model.best,model,optimizer,ep,scheduler,scaler,kf)
        del model,optimizer,loss
        torch.cuda.empty_cache()
def validation(model,val_loader,logger):
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in val_loader:
            data,label= data.to(device), label.to(device,dtype=torch.float)
            output = model(data).squeeze()
            loss = criterion(output,label)
            output = torch.sigmoid(output)
            pred = torch.gt(output, 0.5).long().detach()
            correct += torch.sum(label == pred)
            test_loss += loss.item() 
        acc = correct / len(val_loader.dataset)
        logger.info('Validation: Average Loss: {:.6f}, Accuracy:{}/{}({:.0f}%)\n'.format(
            test_loss/len(val_loader.dataset), correct, len(val_loader.dataset),
        100. * acc))
    return test_loss,acc

def main():
    cfg = get_cfg_defaults()
    cfg.freeze()
    torch.backends.cudnn.benchmark=True
    logger = get_logger(cfg.logger.path, cfg.logger.name)
    logger.info('============ Training routine: "%s" ============\n')
    train(logger,cfg)
    logger.info('=> ============ Network trained - all epochs passed... ============')
    exit()

if __name__ == "__main__":
    main()
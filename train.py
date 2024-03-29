import os
import torch
from torch import nn
from data import get_train_val_loader, get_test_loader, split_dataset
from common.utils import get_device
from common.logger import get_logger
from common.optimizer import build_optimizer, build_scheduler
from common.checkpoint import load_checkpoint, save_checkpoint
from common.configs import get_cfg_defaults
from model.model import get_model, get_ensemble

device = get_device()


def train(
    model,
    logger,
    cfg,
    train_set,
    val_set,
    log_interval=200,
    val_interval=5,
    model_name="swin",
):
    # training
    criterion = nn.BCEWithLogitsLoss()
    acc = 0
    iteration = 0
    best_acc = 0
    max_epoch = cfg.train.epochs
    patience = 5
    last_loss = 1000
    trigger_times = 0
    max_k_folds = cfg.train.k_folds
    for kf in range(0, max_k_folds):
        # get model(model.py)
        logger.info("train " + model_name + " => Loading network architecture...")

        # build optimizer
        logger.info("=> Loading optimizer...")
        optimizer = build_optimizer(cfg, model)

        # build scheduler
        logger.info("=> Loading scheduler...")
        scheduler = build_scheduler(cfg, optimizer)
        scaler = torch.cuda.amp.GradScaler()
        # get dataset(data.py)
        train_loader, val_loader = get_train_val_loader(
            train_set,
            val_set,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            kf=kf,
        )
        # load checkpoint
        model = model.to(device)
        model, optimizer, scheduler, scaler, epoch = load_checkpoint(
            model, optimizer, scheduler, scaler, cfg.model.path[kf], logger, kf
        )
        for ep in range(epoch, max_epoch):
            model.train()
            for batch_idx, (data, label) in enumerate(train_loader):
                data, label = data.to(device), label.to(device, dtype=torch.float)
                optimizer.zero_grad()
                with torch.autocast(
                    device_type=device,
                    dtype=torch.float16,
                    enabled=True,
                    cache_enabled=True,
                ):
                    output = model(data).squeeze()
                    loss = criterion(output, label)

                scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(ep + batch_idx / len(train_loader))
                if iteration % log_interval == 0:
                    logger.info(
                        "Train "
                        + model_name
                        + " Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f},lr:{:.4f}".format(
                            ep,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                            scheduler.get_last_lr()[0],
                        )
                    )
                iteration += 1
                del data, label
                torch.cuda.empty_cache()
            if ep % val_interval == 0:
                test_loss, acc = validation(model, val_loader, logger)
            if test_loss > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    logger.info("Early stopping!\n")
                    return
                last_loss = test_loss
            else:
                trigger_times = 0
                save_checkpoint(
                    cfg.model.path, model, optimizer, ep, scheduler, scaler, kf
                )
                logger.info(f"saving model with loss: {test_loss}")
                last_loss = test_loss
            if acc > best_acc:
                best_acc = acc
                logger.info(f"saving model with acc: {acc}")
                save_checkpoint(
                    cfg.model.best, model, optimizer, ep, scheduler, scaler, kf
                )
        del model, optimizer, loss
        torch.cuda.empty_cache()


def validate_single(data_pack, model, test_loss, correct):
    criterion = nn.BCEWithLogitsLoss()
    data, label = data_pack
    data, label = data.to(device), label.to(device, dtype=torch.float)
    with torch.autocast(
        device_type=device,
        dtype=torch.float16,
        enabled=True,
        cache_enabled=True,
    ):
        output = model(data).squeeze()
        loss = criterion(output, label)
    output = torch.sigmoid(output)
    pred = torch.gt(output, 0.5).long().detach()
    correct += torch.sum(label == pred)
    test_loss += loss.item()


def validation(model, val_loader, logger):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_pack in val_loader:
            validate_single(data_pack, model, test_loss, correct)

        avg_test_loss = test_loss / len(val_loader)
        acc = correct / len(val_loader.dataset)
        logger.info(
            "Validation: Average Loss: {:.6f}, Accuracy:{}/{}({:.0f}%)\n".format(
                avg_test_loss,
                correct,
                len(val_loader.dataset),
                100.0 * acc,
            )
        )
    return avg_test_loss, acc  # Why not return average loss?


def bagging(model, cfg, test_set, logger):
    test_loss = 0
    correct = 0
    test_loader = get_test_loader(test_set)
    with torch.no_grad():
        for data_pack in test_loader:
            model = get_ensemble(cfg.model.path, device, cfg.train.k_folds)
            validate_single(data_pack, model, test_loss, correct)

        avg_test_loss = test_loss / len(test_loader)
        acc = correct / len(test_loader)

        logger.info(
            "Inference: Average Loss: {:.6f}, Accuracy:{}/{}({:.0f}%)\n".format(
                avg_test_loss, correct, len(test_loader.dataset), 100.0 * acc
            )
        )
    return avg_test_loss, acc


def main():
    cfg = get_cfg_defaults()
    cfg.freeze()
    torch.backends.cudnn.benchmark = True
    model = get_model()
    train_set, val_set, test_set = split_dataset(cfg.dataset.path)

    logger = get_logger(cfg.logger.path, cfg.logger.name)
    logger.info('============ Training routine: "%s" ============\n')
    train(model, logger, cfg, train_set, val_set)
    logger.info("=> ============ Network trained - all epochs passed... ============")

    # Inference
    avg_test_loss, acc = bagging(model, cfg, test_set, logger)
    exit()


if __name__ == "__main__":
    main()

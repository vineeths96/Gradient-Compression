import os
import datetime
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist

from model_dispatcher import CIFAR
from reducer import (
    NoneReducer, NoneAllReducer, QSGDReducer, QSGDWECReducer,
    QSGDWECModReducer, TernGradReducer, TernGradModReducer,
    QSGDMaxNormReducer, QSGDBPReducer, QSGDBPAllReducer,
    GlobalRandKMaxNormReducer, MaxNormGlobalRandKReducer,
    NUQSGDModReducer, NUQSGDMaxNormReducer,
    TopKReducer, TopKReducerRatio
)
from timer import Timer
from logger import Logger
from metrics import AverageMeter

config = dict(
    distributed_backend="nccl",
    num_epochs=150,
    batch_size=128,
    architecture="ResNet50",
    # K=10000,
    compression=1/1000,
    reducer="TopKReducerRatio",
    # quantization_level=6,
    seed=42,
    log_verbosity=2,
    lr=0.01,
)


def initiate_distributed():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }

    print(f"[{os.getpid()}] Initializing Process Group with: {env_dict}")
    dist.init_process_group(backend=config['distributed_backend'], init_method="env://")

    print(f"[{os.getpid()}] Initialized Process Group with: RANK = {dist.get_rank()}, "
          + f"WORLD_SIZE = {dist.get_world_size()}" + f", backend={dist.get_backend()}")


def train(local_rank, log_path):
    if local_rank == 0:
        logger = Logger(log_path, config)

    # torch.manual_seed(config["seed"] + local_rank)
    # np.random.seed(config["seed"] + local_rank)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device(f'cuda:{local_rank}')
    timer = Timer(verbosity_level=config["log_verbosity"])

    # reducer = globals()[config['reducer']](device, timer)
    # reducer = globals()[config['reducer']](device, timer, quantization_level=config['quantization_level'])
    # reducer = globals()[config['reducer']](device, timer, K=config['K'], quantization_level=config['quantization_level'])
    # reducer = globals()[config['reducer']](device, timer, K=config['K'])
    reducer = globals()[config['reducer']](device, timer, compression=config['compression'])

    lr = config['lr']
    bits_communicated = 0
    model = CIFAR(device, timer, config['architecture'], config['seed'] + local_rank)

    send_buffers = [torch.zeros_like(param) for param in model.parameters]

    optimizer = optim.SGD(params=model.parameters, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)

    for epoch in range(config['num_epochs']):
        if local_rank == 0:
            logger.log_info("epoch info", {"Progress": epoch / config["num_epochs"], "Current_epoch": epoch}, {"lr": scheduler.get_last_lr()})

        epoch_metrics = AverageMeter(device)

        train_loader = model.train_dataloader(config['batch_size'])
        for i, batch in enumerate(train_loader):
            epoch_frac = epoch + i / model.len_train_loader

            with timer("batch", epoch_frac):
                _, grads, metrics = model.batch_loss_with_gradients(batch)
                epoch_metrics.add(metrics)

                with timer("batch.accumulate", epoch_frac, verbosity=2):
                    for grad, send_buffer in zip(grads, send_buffers):
                        send_buffer[:] = grad

                with timer("batch.reduce", epoch_frac):
                    bits_communicated += reducer.reduce(send_buffers, grads)

                with timer("batch.step", epoch_frac, verbosity=2):
                    optimizer.step()

        scheduler.step()

        with timer("epoch_metrics.collect", epoch, verbosity=2):
            epoch_metrics.reduce()
            if local_rank == 0:
                for key, value in epoch_metrics.values().items():
                    logger.log_info(key, {"value": value, "epoch": epoch, "bits": bits_communicated},
                                    tags={"split": "train"}, )

        with timer("test.last", epoch):
            test_stats = model.test()
            if local_rank == 0:
                for key, value in test_stats.values().items():
                    logger.log_info(key, {"value": value, "epoch": epoch, "bits": bits_communicated},
                                    tags={"split": "test"}, )

        if local_rank == 0:
            logger.epoch_update(epoch, epoch_metrics, test_stats)

    if local_rank == 0:
        print(timer.summary())
        logger.summary_writer(model, timer, bits_communicated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local_world_size', type=int, default=1)
    args = parser.parse_args()
    local_rank = args.local_rank

    log_path = f"./logs/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{config['architecture']}"

    initiate_distributed()
    train(local_rank, log_path)

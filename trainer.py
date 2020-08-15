import os
import datetime
import argparse
import numpy as np
import torch
import torch.distributed as dist

from model_dispatcher import CIFAR
from reducer import (
    NoneReducer, NoneAllReducer, QSGDReducer, QSGDWECReducer, TernGradReducer,
    QSGDWECModReducer, TernGradReducer, TernGradModReducer, QSGDWECMod2Reducer,
    QSGDWECMod3Reducer, QSGDBPReducer, QSGDWECMod4Reducer
)
from timer import Timer
from metrics import AverageMeter


config = dict(
    distributed_backend="nccl",
    num_epochs=150,
    batch_size=128,
    architecture="ResNet50",
    reducer="QSGDWECMod3Reducer",
    quantization_level=8,
    seed=42,
    log_verbosity=2,
    lr=0.01,
)


def log_info(name, values, tags={}):
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)

    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)

    print("{name:20s} - {values} ({tags})".format(name=name, values=values, tags=tags))


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
        os.makedirs(log_path)
        start = datetime.datetime.now()

        metric_list = {'train_top1_accuracy', 'train_top5_accuracy',
                       'test_top1_accuracy', 'test_top5_accuracy',
                       'loss', 'time'}
        log_dict = {metric: np.zeros(config['num_epochs']) for metric in metric_list}

    torch.manual_seed(config["seed"] + local_rank)
    np.random.seed(config["seed"] + local_rank)

    device = torch.device(f'cuda:{local_rank}')
    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=log_info)

    # reducer = globals()[config['reducer']](device, timer)
    reducer = globals()[config['reducer']](device, timer, quantization_level=config['quantization_level'])

    lr = config['lr']
    bits_communicated = 0
    model = CIFAR(device, timer, config['architecture'], config['seed'] + local_rank)

    send_buffers = [torch.zeros_like(param) for param in model.parameters]

    for epoch in range(config['num_epochs']):
        if local_rank == 0:
            log_info("epoch info", {"Progress": epoch / config["num_epochs"], "Current_epoch": epoch})

        epoch_metrics = AverageMeter(device)

        if 0 <= epoch < 50:
            lr = lr
        elif 50 <= epoch < 150:
            lr = lr * 0.1
        elif 250 <= epoch <= 350:
            lr = lr * 0.01
        else:
            lr = 0.0001

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
                # Test paramgrad update
                with timer("batch.step", epoch_frac, verbosity=2):
                    for param, grad in zip(model.parameters, grads):
                        param.data.add_(other=grad, alpha=-lr)

        with timer("epoch_metrics.collect", epoch, verbosity=2):
            epoch_metrics.reduce()
            if local_rank == 0:
                for key, value in epoch_metrics.values().items():
                    log_info(
                        key,
                        {"value": value, "epoch": epoch, "bits": bits_communicated},
                        tags={"split": "train"},
                    )

        with timer("test.last", epoch):
            test_stats = model.test()
            if local_rank == 0:
                for key, value in test_stats.values().items():
                    log_info(
                        key,
                        {"value": value, "epoch": epoch, "bits": bits_communicated},
                        tags={"split": "test"},
                    )

        if local_rank == 0:
            log_dict['train_top1_accuracy'][epoch] = epoch_metrics.values()['top1_accuracy']
            log_dict['train_top5_accuracy'][epoch] = epoch_metrics.values()['top5_accuracy']
            log_dict['test_top1_accuracy'][epoch] = test_stats.values()['top1_accuracy']
            log_dict['test_top5_accuracy'][epoch] = test_stats.values()['top5_accuracy']
            log_dict['loss'][epoch] = epoch_metrics.values()['cross_entropy_loss']
            log_dict['time'][epoch] = (datetime.datetime.now() - start).total_seconds()

    if local_rank == 0:
        print(timer.summary())
        timer.save_summary(f'{log_path}/timer_summary.json')

        with open(f'{log_path}/success.txt', 'w') as file:
            file.write(f"Training completed at {datetime.datetime.now()}\n\n")

            file.write(f"Training parameters\n")
            list_of_strings = [f'{key} : {value}' for key, value in config.items()]
            [file.write(f'{string}\n') for string in list_of_strings]

        np.save(f'{log_path}/log_dict.npy', log_dict)
        torch.save(model.state_dict(), f'{log_path}/model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local_world_size', type=int, default=1)
    args = parser.parse_args()
    local_rank = args.local_rank

    log_path = f"./logs/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{config['architecture']}"

    initiate_distributed()
    train(local_rank, log_path)

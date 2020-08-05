import os
import argparse
import numpy as np
import torch
import torch.distributed as dist

from model_dispatcher import CIFAR
from reducer import (
    NoneReducer, NoneAllReducer, QSGDReducer, QSGDWECReducer, TernGradReducer,
    QSGDWECModReducer, TernGradReducer, TernGradModReducer
)
from timer import Timer
from metrics import AverageMeter


config = dict(
    distributed_backend="nccl",
    num_epochs=350,
    batch_size=128,
    architecture="ResNet50",
    seed=42,
    log_verbosity=2,
    lr=0.01,
)


def log_info(name, values, tags=None):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local_world_size', type=int, default=1)
    args = parser.parse_args()
    local_rank = args.local_rank

    initiate_distributed()
    torch.manual_seed(config["seed"] + args.local_rank)
    np.random.seed(config["seed"] + args.local_rank)

    device = torch.device(f'cuda:{args.local_rank}')
    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=log_info)

    # reducer = NoneReducer(device, timer)
    # reducer = NoneAllReducer(device, timer)
    # reducer = QSGDReducer(device, timer, quantization_level=8)
    reducer = QSGDWECModReducer(device, timer, quantization_level=8)
    # reducer = TernGradModReducer(device, timer)

    lr = config['lr']
    bits_communicated = 0
    model = CIFAR(device, timer, config['architecture'], config['seed'] + args.local_rank)

    send_buffers = [torch.zeros_like(param) for param in model.parameters]

    for epoch in range(config['num_epochs']):
        print("Epoch", epoch)
        epoch_metrics = AverageMeter(device)

        if 0 <= epoch < 150:
            lr = lr
        elif 150 <= epoch < 250:
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
                for key, value in test_stats.items():
                    log_info(
                        key,
                        {"value": value, "epoch": epoch, "bits": bits_communicated},
                        tags={"split": "test"},
                    )

    if local_rank == 0:
        print(timer.summary())
        timer.save_summary(os.path.join("timer_summary.json"))

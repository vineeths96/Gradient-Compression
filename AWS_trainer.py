import os
import datetime
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist

from model_dispatcher import CIFAR
from reducer import (
    NoneReducer,
    NoneAllReducer,
    QSGDReducer,
    QSGDWECReducer,
    QSGDWECModReducer,
    TernGradReducer,
    TernGradModReducer,
    QSGDMaxNormReducer,
    # QSGDBPReducer,
    # QSGDBPAllReducer,
    GlobalRandKMaxNormReducer,
    MaxNormGlobalRandKReducer,
    NUQSGDModReducer,
    NUQSGDMaxNormReducer,
    TopKReducer,
    TopKReducerRatio,
    GlobalTopKReducer,
    GlobalTopKReducerRatio,
    QSGDMaxNormBiasedReducer,
    QSGDMaxNormBiasedMemoryReducer,
    NUQSGDMaxNormBiasedReducer,
    NUQSGDMaxNormBiasedMemoryReducer,
    QSGDMaxNormTwoScaleReducer,
    GlobalRandKMaxNormTwoScaleReducer,
)
from timer import Timer
from logger import Logger
from metrics import AverageMeter


def initiate_distributed():
    env_dict = {key: os.environ[key] for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")}

    print(f"[{os.getpid()}] Initializing Process Group with: {env_dict}")
    dist.init_process_group(backend='nccl', init_method="env://")

    print(
        f"[{os.getpid()}] Initialized Process Group with: RANK = {dist.get_rank()}, "
        + f"WORLD_SIZE = {dist.get_world_size()}"
        + f", backend={dist.get_backend()}"
    )


def train(local_rank):
    local_step_epoch = {"H_1": 1, "H_5": 1, "H_10": 2, "H_15": 3, "H_20": 4}

    for local_step in local_step_epoch.keys():
        for model_name in ['ResNet50', 'VGG16']:
            for reducer in ['NoneAllReducer', 'QSGDMaxNormReducer', 'GlobalRandKMaxNormReducer', 'QSGDMaxNormTwoScaleReducer', 'GlobalRandKMaxNormTwoScaleReducer']:
                import time
                time.sleep(75)

                config = dict(
                    distributed_backend="nccl",
                    num_epochs=local_step_epoch[local_step],
                    batch_size=128,
                    architecture=model_name,
                    local_steps=int(local_step.split('_')[-1]),
                    reducer=reducer,
                    seed=42,
                    log_verbosity=2,
                    lr=0.01,
                )

                log_path = f"./logs/{local_step}/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_{config['architecture']}"

                logger = Logger(log_path, config, local_rank)

                # torch.manual_seed(config["seed"] + local_rank)
                # np.random.seed(config["seed"] + local_rank)
                torch.manual_seed(config["seed"])
                np.random.seed(config["seed"])

                device = torch.device(f"cuda:{local_rank}")
                timer = Timer(verbosity_level=config["log_verbosity"])

                if config["reducer"] in [
                    "NoneReducer",
                    "NoneAllReducer",
                    "TernGradReducer",
                    "TernGradModReducer",
                ]:
                    reducer = globals()[config["reducer"]](device, timer)
                elif config["reducer"] in [
                    "QSGDReducer",
                    "QSGDWECReducer",
                    "QSGDWECModReducer",
                    "QSGDBPReducer",
                    "QSGDBPAllReducer",
                    "QSGDMaxNormReducer",
                    "NUQSGDModReducer",
                    "NUQSGDMaxNormReducer",
                    "QSGDMaxNormBiasedReducer",
                    "QSGDMaxNormBiasedMemoryReducer",
                    "NUQSGDMaxNormBiasedReducer",
                    "NUQSGDMaxNormBiasedMemoryReducer",
                    "QSGDMaxNormMaskReducer",
                ]:
                    config['quantization_level'] = 6
                    reducer = globals()[config["reducer"]](device, timer,
                                                           quantization_level=config["quantization_level"])
                elif config["reducer"] in [
                    "GlobalRandKMaxNormReducer",
                    "MaxNormGlobalRandKReducer",
                ]:
                    config['K'] = 10000
                    config['quantization_level'] = 6
                    reducer = globals()[config["reducer"]](
                        device,
                        timer,
                        K=config["K"],
                        quantization_level=config["quantization_level"],
                    )
                elif config["reducer"] in ["TopKReducer", "GlobalTopKReducer"]:
                    config['K'] = 10000
                    reducer = globals()[config["reducer"]](device, timer, K=config["K"])
                elif config["reducer"] in ["TopKReducerRatio", "GlobalTopKReducerRatio"]:
                    config["compression"] = 0.001
                    reducer = globals()[config["reducer"]](device, timer, compression=config["compression"])
                elif config["reducer"] in ["QSGDMaxNormTwoScaleReducer"]:
                    config['quantization_level'] = 6
                    config['higher_quantization_level'] = 10
                    reducer = globals()[config["reducer"]](
                        device,
                        timer,
                        lower_quantization_level=config["quantization_level"],
                        higher_quantization_level=config["higher_quantization_level"],
                    )
                elif config["reducer"] in ["GlobalRandKMaxNormTwoScaleReducer"]:
                    config['K'] = 10000
                    config['quantization_level'] = 6
                    config['higher_quantization_level'] = 10
                    reducer = globals()[config["reducer"]](
                        device,
                        timer,
                        lower_quantization_level=config["quantization_level"],
                        higher_quantization_level=config["higher_quantization_level"],
                    )
                else:
                    raise NotImplementedError("Reducer method not implemented")

                lr = config["lr"]
                bits_communicated = 0

                global_iteration_count = 0
                model = CIFAR(device, timer, config["architecture"], config["seed"] + local_rank)

                send_buffers = [torch.zeros_like(param) for param in model.parameters]

                optimizer = optim.SGD(params=model.parameters, lr=lr, momentum=0.9)
                scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)

                for epoch in range(config["num_epochs"]):
                    if local_rank == 0:
                        logger.log_info(
                            "epoch info",
                            {"Progress": epoch / config["num_epochs"], "Current_epoch": epoch},
                            {"lr": scheduler.get_last_lr()},
                        )

                    epoch_metrics = AverageMeter(device)

                    train_loader = model.train_dataloader(config["batch_size"])
                    for i, batch in enumerate(train_loader):
                        global_iteration_count += 1
                        epoch_frac = epoch + i / model.len_train_loader

                        with timer("batch", epoch_frac):
                            _, grads, metrics = model.batch_loss_with_gradients(batch)
                            epoch_metrics.add(metrics)

                            if global_iteration_count % config["local_steps"] == 0:
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
                                logger.log_info(
                                    key,
                                    {"value": value, "epoch": epoch, "bits": bits_communicated},
                                    tags={"split": "train"},
                                )

                    with timer("test.last", epoch):
                        test_stats = model.test()
                        if local_rank == 0:
                            for key, value in test_stats.values().items():
                                logger.log_info(
                                    key,
                                    {"value": value, "epoch": epoch, "bits": bits_communicated},
                                    tags={"split": "test"},
                                )

                    if local_rank == 0:
                        logger.epoch_update(epoch, epoch_metrics, test_stats)

                if local_rank == 0:
                    print(timer.summary())

                logger.summary_writer(model, timer, bits_communicated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    local_rank = args.local_rank

    initiate_distributed()
    train(local_rank)
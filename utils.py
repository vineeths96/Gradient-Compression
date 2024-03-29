import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    TransformedBbox,
    BboxPatch,
    BboxConnector,
)


NUM_REPEATS = 5
label_dict = {
    "NoneAllReducer": "AllReduce SGD",
    "QSGDMaxNormReducer": "QSGD-MN",
    "GlobalRandKMaxNormReducer": "GRandK-MN",
    "QSGDMaxNormTwoScaleReducer": "QSGD-MN-TS",
    "GlobalRandKMaxNormTwoScaleReducer": "GRandK-MN-TS",
    "RankKReducer": "PowerSGD",
}


def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2


def plot_loss_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None
            rank = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = label_dict[line.split(": ")[-1]]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

                    if line.startswith("rank"):
                        rank = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, f"({quant_level},{higher_quant_level})", "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            elif rank:
                label = " ".join([reducer, "Rank", rank])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            loss = log_dict[()].get("train_loss")
            num_epochs = loss.shape[1]

            mean_loss = np.mean(loss, axis=0)
            std_dev_loss = np.std(loss, axis=0)

            axes_main.plot(np.arange(num_epochs), mean_loss, label=label)
            axes_main.fill_between(
                np.arange(num_epochs),
                mean_loss - std_dev_loss,
                mean_loss + std_dev_loss,
                alpha=0.25,
            )
            axes_main.set_ylim(0, 2.5)

            # axes_inner.plot(axes_inner_range, mean_loss[axes_inner_range])
            # axes_inner.fill_between(
            #     axes_inner_range,
            #     mean_loss[axes_inner_range] - std_dev_loss[axes_inner_range],
            #     mean_loss[axes_inner_range] + std_dev_loss[axes_inner_range],
            #     alpha=0,
            # )
            # axes_inner.set_ylim(0, 2.5)

            # axes_main.plot(loss, label=label)
            # axes_inner.plot(axes_inner_range, mean_loss[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Loss")
        # axes_main.set_title(f"Loss curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/loss_{models[group_ind]}.svg")
        plt.show()


def plot_loss_time_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None
            rank = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = label_dict[line.split(": ")[-1]]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

                    if line.startswith("rank"):
                        rank = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, f"({quant_level},{higher_quant_level})", "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            elif rank:
                label = " ".join([reducer, "Rank", rank])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            loss = log_dict[()].get("train_loss")
            time = log_dict[()].get("time")

            num_epochs = loss.shape[1]

            for i in reversed(range(1, NUM_REPEATS)):
                time[i] -= time[i - 1][-1]

            mean_loss = np.mean(loss, axis=0)
            std_dev_loss = np.std(loss, axis=0)
            time = np.mean(time, axis=0)

            axes_main.plot(time, mean_loss, label=label)
            axes_main.fill_between(
                time,
                mean_loss - std_dev_loss,
                mean_loss + std_dev_loss,
                alpha=0.25,
            )
            axes_main.set_ylim(0, 2.5)

            # axes_inner.plot(time[axes_inner_range], mean_loss[axes_inner_range])
            # axes_inner.fill_between(
            #     time[axes_inner_range],
            #     mean_loss[axes_inner_range] - std_dev_loss[axes_inner_range],
            #     mean_loss[axes_inner_range] + std_dev_loss[axes_inner_range],
            #     alpha=0,
            # )
            # axes_inner.set_ylim(0, 2.5)

            # axes_main.plot(time, loss, label=label)
            # axes_inner.plot(time[axes_inner_range], loss[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Time")
        axes_main.set_ylabel("Loss")
        # axes_main.set_title(f"Loss Time curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/loss_time_{models[group_ind]}.svg")
        plt.show()


def plot_top1_accuracy_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None
            rank = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = label_dict[line.split(": ")[-1]]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

                    if line.startswith("rank"):
                        rank = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, f"({quant_level},{higher_quant_level})", "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            elif rank:
                label = " ".join([reducer, "Rank", rank])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top1_accuracy = log_dict[()].get("test_top1_accuracy") * 100
            num_epochs = top1_accuracy.shape[1]

            mean_top1_accuracy = np.mean(top1_accuracy, axis=0)
            std_dev_top1_accuracy = np.std(top1_accuracy, axis=0)

            axes_main.plot(np.arange(num_epochs), mean_top1_accuracy, label=label)
            axes_main.fill_between(
                np.arange(num_epochs),
                mean_top1_accuracy - std_dev_top1_accuracy,
                mean_top1_accuracy + std_dev_top1_accuracy,
                alpha=0.25,
            )

            # axes_inner.plot(axes_inner_range, mean_top1_accuracy[axes_inner_range])
            # axes_inner.fill_between(
            #     axes_inner_range,
            #     mean_top1_accuracy[axes_inner_range] - std_dev_top1_accuracy[axes_inner_range],
            #     mean_top1_accuracy[axes_inner_range] + std_dev_top1_accuracy[axes_inner_range],
            #     alpha=0,
            # )

            # axes_main.plot(top1_accuracy, label=label)
            # axes_inner.plot(axes_inner_range, top1_accuracy[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Test Accuracy (%)")
        # axes_main.set_title(f"Accuracy curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top1_{models[group_ind]}.svg")
        plt.show()


def plot_top5_accuracy_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None
            rank = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = label_dict[line.split(": ")[-1]]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

                    if line.startswith("rank"):
                        rank = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, f"({quant_level},{higher_quant_level})", "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            elif rank:
                label = " ".join([reducer, "Rank", rank])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top5_accuracy = log_dict[()].get("test_top5_accuracy") * 100
            num_epochs = top5_accuracy.shape[1]

            mean_top5_accuracy = np.mean(top5_accuracy, axis=0)
            std_dev_top5_accuracy = np.std(top5_accuracy, axis=0)

            axes_main.plot(np.arange(num_epochs), mean_top5_accuracy, label=label)
            axes_main.fill_between(
                np.arange(num_epochs),
                mean_top5_accuracy - std_dev_top5_accuracy,
                mean_top5_accuracy + std_dev_top5_accuracy,
                alpha=0.25,
            )

            # axes_inner.plot(axes_inner_range, mean_top5_accuracy[axes_inner_range])
            # axes_inner.fill_between(
            #     axes_inner_range,
            #     mean_top5_accuracy[axes_inner_range] - std_dev_top5_accuracy[axes_inner_range],
            #     mean_top5_accuracy[axes_inner_range] + std_dev_top5_accuracy[axes_inner_range],
            #     alpha=0,
            # )

            # axes_main.plot(top5_accuracy, label=label)
            # axes_inner.plot(axes_inner_range, top5_accuracy[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Test Accuracy (%)")
        # axes_main.set_title(f"Accuracy curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top5_{models[group_ind]}.svg")
        plt.show()


def plot_top1_accuracy_time_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None
            rank = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = label_dict[line.split(": ")[-1]]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

                    if line.startswith("rank"):
                        rank = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, f"({quant_level},{higher_quant_level})", "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            elif rank:
                label = " ".join([reducer, "Rank", rank])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top1_accuracy = log_dict[()].get("test_top1_accuracy") * 100
            time = log_dict[()].get("time")

            num_epochs = top1_accuracy.shape[1]

            for i in reversed(range(1, NUM_REPEATS)):
                time[i] -= time[i - 1][-1]

            mean_top1_accuracy = np.mean(top1_accuracy, axis=0)
            std_dev_top1_accuracy = np.std(top1_accuracy, axis=0)
            time = np.mean(time, axis=0)

            axes_main.plot(time, mean_top1_accuracy, label=label)
            axes_main.fill_between(
                time,
                mean_top1_accuracy - std_dev_top1_accuracy,
                mean_top1_accuracy + std_dev_top1_accuracy,
                alpha=0.25,
            )

            # axes_inner.plot(time[axes_inner_range], mean_top1_accuracy[axes_inner_range])
            # axes_inner.fill_between(
            #     time[axes_inner_range],
            #     mean_top1_accuracy[axes_inner_range] - std_dev_top1_accuracy[axes_inner_range],
            #     mean_top1_accuracy[axes_inner_range] + std_dev_top1_accuracy[axes_inner_range],
            #     alpha=0,
            # )

            # axes_main.plot(time, top1_accuracy, label=label)
            # axes_inner.plot(time[axes_inner_range], top1_accuracy[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Time (sec)")
        axes_main.set_ylabel("Test Accuracy (%)")
        # axes_main.set_title(f"Accuracy Time curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top1_time_{models[group_ind]}.svg")
        plt.show()


def plot_time_breakdown(log_path):
    time_labels = [
        "batch",
        # "batch.accumulate",
        "batch.forward",
        "batch.backward",
        "batch.reduce",
        "batch.evaluate",
        "batch.step",
    ]

    models = ["ResNet50", "VGG16"]

    [plt.figure(num=ind) for ind in range(len(models))]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    events = np.arange(len(time_labels))
    width = 0.1

    for group_ind, experiment_group in enumerate(experiment_groups):
        plt.figure(num=group_ind)
        experiment_group.sort()

        num_experiments = len(experiment_group) - 1

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None
            rank = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = label_dict[line.split(": ")[-1]]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

                    if line.startswith("rank"):
                        rank = line.split(": ")[-1]

                if higher_quant_level:
                    label = " ".join([reducer, f"({quant_level},{higher_quant_level})", "bits"])
                elif quant_level:
                    label = " ".join([reducer, quant_level, "bits"])
                elif compression:
                    label = " ".join([reducer, "K:", compression])
                elif rank:
                    label = " ".join([reducer, "Rank", rank])
                else:
                    label = reducer

            time_df = pd.read_json(os.path.join(experiment, "timer_summary_0.json")).loc["average_duration"]
            time_values = time_df[time_labels].values

            plt.bar(
                events + (ind - num_experiments / 2) * width,
                time_values,
                width,
                label=label,
            )

        # plt.grid()
        time_labels_axis = [time_label.split(".")[-1] for time_label in time_labels]
        plt.xticks(events, time_labels_axis)
        plt.ylabel("Average Time (sec)")
        # plt.title(f"Time breakdown {models[group_ind]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./plots/time_breakdown_{models[group_ind]}.svg")
    plt.show()


def plot_time_scalability(log_path):
    time_labels = ["batch"]
    models = ["ResNet50", "VGG16"]
    # instances = ["P2", "P3", "P2 Multi Node", "P3 Multi Node"]
    instances = ["P3", "P3 Multi Node"]

    for instance in instances:
        GPUs = os.listdir(os.path.join(log_path, instance))
        GPUs.sort()

        width = 0.1
        events = np.arange(len(GPUs))

        time_dfs = {model: None for model in models}
        experiment_groups = [glob.glob(f"{log_path}/{instance}/*/*{model}") for model in models]

        for group_ind, experiment_group in enumerate(experiment_groups):
            time_results = []
            compressor_ind_map = {}
            latest_compressor_ind = 0

            experiment_group.sort()

            for ind, experiment in enumerate(experiment_group):
                reducer = None
                quant_level = None
                higher_quant_level = None
                compression = None
                rank = None
                num_epochs = None

                with open(os.path.join(experiment, "success.txt")) as file:
                    for line in file:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = label_dict[line.split(": ")[-1]]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(": ")[-1]

                        if line.startswith("higher_quantization_level"):
                            higher_quant_level = line.split(": ")[-1]

                        if line.startswith("compression"):
                            compression = line.split(": ")[-1]

                        if line.startswith("num_epochs"):
                            num_epochs = int(line.split(": ")[-1])

                        if line.startswith("rank"):
                            rank = line.split(": ")[-1]

                    if higher_quant_level:
                        label = " ".join([reducer, f"({quant_level},{higher_quant_level})", "bits"])
                    elif quant_level:
                        label = " ".join([reducer, quant_level, "bits"])
                    elif compression:
                        label = " ".join([reducer, "K:", compression])
                    elif rank:
                        label = " ".join([reducer, "Rank", rank])
                    else:
                        label = reducer

                if not label in compressor_ind_map:
                    time_results.append([])
                    compressor_ind_map[label] = latest_compressor_ind
                    latest_compressor_ind += 1

                time_df = pd.read_json(os.path.join(experiment, "timer_summary.json")).loc["total_time"]
                time_values = time_df[time_labels].values / num_epochs

                time_results[compressor_ind_map[label]].append(float(time_values))

            time_dfs[models[group_ind]] = pd.DataFrame(time_results, index=compressor_ind_map.keys())

        for df_key in time_dfs:
            plt.figure()
            time_df = time_dfs[df_key]
            num_compressors = len(time_df) - 1

            for ind, (label, values) in enumerate(time_df.iterrows()):
                values = values.to_list()
                plt.bar(
                    events + (ind - num_compressors / 2) * width,
                    values,
                    width,
                    label=label,
                )

            # plt.grid()
            plt.xticks(events, GPUs)
            plt.ylabel("Time per Epoch (sec)")
            # plt.title(f"Time Scalability {df_key} {instance}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./plots/time_scalability_{df_key}_{instance}.svg")
            plt.show()


def plot_throughput_scalability(log_path):
    time_labels = ["batch"]
    models = ["ResNet50", "VGG16"]
    # instances = ["P2", "P3", "P2 Multi Node", "P3 Multi Node"]
    instances = ["P3", "P3 Multi Node"]

    for instance in instances:
        GPUs = os.listdir(os.path.join(log_path, instance))
        GPUs.sort()

        width = 0.1
        events = np.arange(len(GPUs))

        throughput_dfs = {model: None for model in models}
        experiment_groups = [glob.glob(f"{log_path}/{instance}/*/*{model}") for model in models]

        for group_ind, experiment_group in enumerate(experiment_groups):
            throughput_results = []
            compressor_ind_map = {}
            latest_compressor_ind = 0

            experiment_group.sort()

            for ind, experiment in enumerate(experiment_group):
                reducer = None
                quant_level = None
                higher_quant_level = None
                compression = None
                rank = None

                with open(os.path.join(experiment, "success.txt")) as file:
                    for line in file:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = label_dict[line.split(": ")[-1]]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(": ")[-1]

                        if line.startswith("higher_quantization_level"):
                            higher_quant_level = line.split(": ")[-1]

                        if line.startswith("compression"):
                            compression = line.split(": ")[-1]

                    if line.startswith("rank"):
                        rank = line.split(": ")[-1]

                    if higher_quant_level:
                        label = " ".join([reducer, f"({quant_level},{higher_quant_level})", "bits"])
                    elif quant_level:
                        label = " ".join([reducer, quant_level, "bits"])
                    elif compression:
                        label = " ".join([reducer, "K:", compression])
                    elif rank:
                        label = " ".join([reducer, "Rank", rank])
                    else:
                        label = reducer

                if not label in compressor_ind_map:
                    throughput_results.append([])
                    compressor_ind_map[label] = latest_compressor_ind
                    latest_compressor_ind += 1

                time_df = pd.read_json(os.path.join(experiment, "timer_summary.json")).loc["average_duration"]
                num_GPUs = int(experiment.split("/")[5].split()[0])
                throughput = (128 * num_GPUs) / time_df[time_labels].values

                throughput_results[compressor_ind_map[label]].append(int(throughput))

            throughput_dfs[models[group_ind]] = pd.DataFrame(throughput_results, index=compressor_ind_map.keys())

        for df_key in throughput_dfs:
            plt.figure()
            throughput_df = throughput_dfs[df_key]
            num_compressors = len(throughput_df) - 1

            for ind, (label, values) in enumerate(throughput_df.iterrows()):
                values = values.to_list()
                plt.bar(
                    events + (ind - num_compressors / 2) * width,
                    values,
                    width,
                    label=label,
                )

            # plt.grid()
            plt.xticks(events, GPUs)
            plt.ylabel("Images per sec")
            # plt.title(f"Throughput Scalability {df_key} {instance}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./plots/throughput_scalability_{df_key}_{instance}.svg")
            plt.show()


def plot_performance_modelling(log_path):
    models = ["ResNet50", "VGG16"]
    instances = ["P3"]  # , "P3 Multi Node"]

    batch_size = 128
    inter_gpu_bw = 200 * 1024
    gpu_cpu_bw = 11 * 1024
    network_latency = 9e-3
    network_bw = 1 * 1024 / 8
    num_gpu_per_node = 4

    for instance in instances:
        GPUs = [1, 2, 4, 8, 16, 32, 64, 128]

        width = 0.1
        events = np.arange(len(GPUs))

        throughput_dfs = {model: None for model in models}
        experiment_groups = [glob.glob(f"{log_path}/{instance}/1 GPU/*{model}") for model in models]

        for group_ind, experiment_group in enumerate(experiment_groups):
            throughput_results = []
            compressor_ind_map = {}
            latest_compressor_ind = 0

            experiment_group.sort()

            for ind, experiment in enumerate(experiment_group):
                reducer = None
                quant_level = None
                higher_quant_level = None
                compression = None
                rank = None

                with open(os.path.join(experiment, "success.txt")) as file:
                    for line in file:
                        line = line.rstrip()

                        if line.startswith("architecture"):
                            architecture = line.split(": ")[-1]

                        if line.startswith("reducer"):
                            reducer = label_dict[line.split(": ")[-1]]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(": ")[-1]

                        if line.startswith("higher_quantization_level"):
                            higher_quant_level = line.split(": ")[-1]

                        if line.startswith("compression"):
                            compression = line.split(": ")[-1]

                        if line.startswith("rank"):
                            rank = line.split(": ")[-1]

                    if higher_quant_level:
                        # label = " ".join([reducer, f"({quant_level},{higher_quant_level})", "bits"])
                        label = " ".join([reducer])
                    elif quant_level:
                        # label = " ".join([reducer, quant_level, "bits"])
                        label = " ".join([reducer])
                    elif compression:
                        label = " ".join([reducer, "K:", compression])
                    elif rank:
                        label = " ".join([reducer, "Rank", rank])
                    else:
                        label = reducer

                if not label in compressor_ind_map:
                    throughput_results.append([])
                    compressor_ind_map[label] = latest_compressor_ind
                    latest_compressor_ind += 1

                time_df = pd.read_json(os.path.join(experiment, "timer_summary.json"))

                if architecture == "ResNet50":
                    gradient_size = 89.72
                elif architecture == "VGG16":
                    gradient_size = 56.18
                else:
                    raise ValueError("Model undefined")

                if label_dict["NoneAllReducer"] in label:
                    gradient_size = gradient_size
                elif label_dict["QSGDMaxNormReducer"] in label:
                    gradient_size /= 4
                elif label_dict["QSGDMaxNormTwoScaleReducer"] in label:
                    gradient_size /= 2
                elif label_dict["GlobalRandKMaxNormReducer"] in label:
                    if architecture == "ResNet50":
                        gradient_size *= 10000 / 23520842
                    elif architecture == "VGG16":
                        gradient_size *= 10000 / 14728266
                    gradient_size /= 4
                elif label_dict["GlobalRandKMaxNormTwoScaleReducer"] in label:
                    if architecture == "ResNet50":
                        gradient_size *= 10000 / 23520842
                    elif architecture == "VGG16":
                        gradient_size *= 10000 / 14728266
                    gradient_size /= 2
                else:
                    raise ValueError("Method undefined")

                for gpu in GPUs:
                    if gpu > num_gpu_per_node:
                        num_nodes = gpu / num_gpu_per_node
                    else:
                        num_nodes = 1

                    T = (
                        time_df["batch"]["average_duration"]
                        + gradient_size / inter_gpu_bw * np.log2(num_gpu_per_node)
                        + (network_latency + gradient_size / network_bw) * np.log2(num_nodes)
                    )
                    throughput = (batch_size * gpu) / T

                    throughput_results[compressor_ind_map[label]].append(int(throughput))

            throughput_dfs[models[group_ind]] = pd.DataFrame(throughput_results, index=compressor_ind_map.keys())

        for df_key in throughput_dfs:
            fig, axes_main = plt.subplots()
            axes_inner = plt.axes([0.25, 0.35, 0.3, 0.3])
            axes_inner_range = list(range(0, 4))

            throughput_df = throughput_dfs[df_key]
            num_compressors = len(throughput_df) - 1

            for ind, (label, values) in enumerate(throughput_df.iterrows()):
                values = values.to_list()
                axes_main.bar(
                    events + (ind - num_compressors / 2) * width,
                    values,
                    width,
                    label=label,
                )

            INNER_GPUs = 5
            for ind, (label, values) in enumerate(throughput_df.iterrows()):
                axes_inner.bar(
                    events[:INNER_GPUs] + (ind - num_compressors / 2) * width,
                    values[:INNER_GPUs],
                    width,
                    label=label,
                )
            # axes_inner.grid()
            axes_inner.set_xticks(events[:INNER_GPUs])
            axes_inner.set_xticklabels(GPUs[:INNER_GPUs])
            mark_inset(
                axes_main,
                axes_inner,
                loc1a=4,
                loc1b=1,
                loc2a=3,
                loc2b=2,
                fc="none",
                ec="0.5",
            )

            # axes_main.grid()
            axes_main.set_xticks(events)
            axes_main.set_xticklabels(GPUs)
            axes_main.set_ylabel("Images per sec")
            axes_main.set_xlabel("Number of GPUs")
            # axes_main.set_title(f"Performance Modelling {df_key} {instance}")
            axes_main.legend()

            plt.tight_layout()
            plt.savefig(f"./plots/performance_modelling_{df_key}_{instance}.svg")
            plt.show()


if __name__ == "__main__":
    root_log_path = "./logs/plot_logs/"

    plot_loss_curves(os.path.join(root_log_path, "convergence"))
    plot_loss_time_curves(os.path.join(root_log_path, "convergence"))
    plot_top1_accuracy_curves(os.path.join(root_log_path, "convergence"))
    plot_top1_accuracy_time_curves(os.path.join(root_log_path, "convergence"))
    plot_top5_accuracy_curves(os.path.join(root_log_path, "convergence"))

    plot_time_breakdown(os.path.join(root_log_path, "time_breakdown"))
    plot_time_scalability(os.path.join(root_log_path, "scalability"))
    plot_throughput_scalability(os.path.join(root_log_path, "scalability"))

    plot_performance_modelling(os.path.join(root_log_path, "scalability"))

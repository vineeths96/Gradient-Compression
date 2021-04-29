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
        fig, axes_main = plt.subplots(figsize=[10, 7])
        axes_inner = plt.axes([0.25, 0.6, 0.3, 0.3])
        axes_inner_range = list(range(40, 80))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = line.split(": ")[-1]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            loss = log_dict[()].get("test_loss")
            axes_main.plot(loss, label=label)

            axes_inner.plot(axes_inner_range, loss[axes_inner_range])

        axes_inner.grid()
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
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Loss")
        axes_main.set_title(f"Loss curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/loss_{models[group_ind]}.png")
        plt.show()


def plot_loss_time_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots(figsize=[10, 7])
        axes_inner = plt.axes([0.25, 0.6, 0.3, 0.3])
        axes_inner_range = list(range(40, 80))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = line.split(": ")[-1]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            loss = log_dict[()].get("test_loss")
            time = log_dict[()].get("time")
            axes_main.plot(time, loss, label=label)

            axes_inner.plot(time[axes_inner_range], loss[axes_inner_range])

        axes_inner.grid()
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
        axes_main.set_xlabel("TIme")
        axes_main.set_ylabel("Loss")
        axes_main.set_title(f"Loss Time curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/loss_time_{models[group_ind]}.png")
        plt.show()


def plot_top1_accuracy_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots(figsize=[10, 7])
        axes_inner = plt.axes([0.25, 0.15, 0.3, 0.3])
        axes_inner_range = list(range(30, 60))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = line.split(": ")[-1]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top1_accuracy = log_dict[()].get("test_top1_accuracy")
            axes_main.plot(top1_accuracy, label=label)

            axes_inner.plot(axes_inner_range, top1_accuracy[axes_inner_range])

        axes_inner.grid()
        mark_inset(
            axes_main,
            axes_inner,
            loc1a=1,
            loc1b=4,
            loc2a=2,
            loc2b=3,
            fc="none",
            ec="0.5",
        )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Top 1 Accuracy")
        axes_main.set_title(f"Accuracy curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top1_{models[group_ind]}.png")
        plt.show()


def plot_top5_accuracy_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots(figsize=[10, 7])
        axes_inner = plt.axes([0.25, 0.15, 0.3, 0.3])
        axes_inner_range = list(range(5, 25))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = line.split(": ")[-1]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top5_accuracy = log_dict[()].get("test_top5_accuracy")
            axes_main.plot(top5_accuracy, label=label)

            axes_inner.plot(axes_inner_range, top5_accuracy[axes_inner_range])

        axes_inner.grid()
        mark_inset(
            axes_main,
            axes_inner,
            loc1a=1,
            loc1b=4,
            loc2a=2,
            loc2b=3,
            fc="none",
            ec="0.5",
        )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Top 5 Accuracy")
        axes_main.set_title(f"Accuracy curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top5_{models[group_ind]}.png")
        plt.show()


def plot_top1_accuracy_time_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots(figsize=[10, 7])
        axes_inner = plt.axes([0.25, 0.15, 0.3, 0.3])
        axes_inner_range = list(range(30, 60))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = line.split(": ")[-1]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

            if higher_quant_level:
                label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
            elif quant_level:
                label = " ".join([reducer, quant_level, "bits"])
            elif compression:
                label = " ".join([reducer, "K:", compression])
            else:
                label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top1_accuracy = log_dict[()].get("test_top1_accuracy")
            time = log_dict[()].get("time")
            axes_main.plot(time, top1_accuracy, label=label)

            axes_inner.plot(time[axes_inner_range], top1_accuracy[axes_inner_range])

        axes_inner.grid()
        mark_inset(
            axes_main,
            axes_inner,
            loc1a=1,
            loc1b=4,
            loc2a=2,
            loc2b=3,
            fc="none",
            ec="0.5",
        )

        # axes_main.grid()
        axes_main.set_xlabel("Time")
        axes_main.set_ylabel("Top 1 Accuracy")
        axes_main.set_title(f"Accuracy Time curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top1_time_{models[group_ind]}.png")
        plt.show()


def plot_time_per_batch_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        plt.figure(figsize=[10, 7])

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = line.split(": ")[-1]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

                if higher_quant_level:
                    label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
                elif quant_level:
                    label = " ".join([reducer, quant_level, "bits"])
                elif compression:
                    label = " ".join([reducer, "K:", compression])
                else:
                    label = reducer

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            time = log_dict[()].get("time")
            epoch_time = np.zeros(len(time) - 1)

            for ind in range(epoch_time.shape[0]):
                epoch_time[ind] = time[ind + 1] - time[ind]

            plt.plot(epoch_time, label=label)

        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel("Average time")
        plt.title(f"Average time curve {models[group_ind]}")
        plt.legend()
        plt.savefig(f"./plots/time_{models[group_ind]}.png")
        plt.show()


def plot_time_breakdown(log_path):
    time_labels = [
        "batch",
        "batch.accumulate",
        "batch.backward",
        "batch.evaluate",
        "batch.forward",
        "batch.reduce",
        "batch.step",
    ]

    models = ["ResNet50", "VGG16"]

    [plt.figure(num=ind, figsize=[10, 7]) for ind in range(len(models))]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    events = np.arange(len(time_labels))
    width = 0.15

    for group_ind, experiment_group in enumerate(experiment_groups):
        plt.figure(num=group_ind)
        experiment_group.sort()

        num_experiments = len(experiment_group) - 1

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            higher_quant_level = None
            compression = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = line.split(": ")[-1]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(": ")[-1]

                    if line.startswith("higher_quantization_level"):
                        higher_quant_level = line.split(": ")[-1]

                    if line.startswith("compression"):
                        compression = line.split(": ")[-1]

                if higher_quant_level:
                    label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
                elif quant_level:
                    label = " ".join([reducer, quant_level, "bits"])
                elif compression:
                    label = " ".join([reducer, "K:", compression])
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

        plt.grid()
        plt.xticks(events, time_labels)
        plt.ylabel("Average time")
        plt.title(f"Time breakdown {models[group_ind]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./plots/time_breakdown_{models[group_ind]}.png")
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
                num_epochs = None

                with open(os.path.join(experiment, "success.txt")) as file:
                    for line in file:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = line.split(": ")[-1]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(": ")[-1]

                        if line.startswith("higher_quantization_level"):
                            higher_quant_level = line.split(": ")[-1]

                        if line.startswith("compression"):
                            compression = line.split(": ")[-1]

                        if line.startswith("num_epochs"):
                            num_epochs = int(line.split(": ")[-1])

                    if higher_quant_level:
                        label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
                    elif quant_level:
                        label = " ".join([reducer, quant_level, "bits"])
                    elif compression:
                        label = " ".join([reducer, "K:", compression])
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
            plt.figure(figsize=[10, 7])
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

            plt.grid()
            plt.xticks(events, GPUs)
            plt.ylabel("Time per epoch")
            plt.title(f"Time Scalability {df_key} {instance}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./plots/time_scalability_{df_key}_{instance}.png")

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

                with open(os.path.join(experiment, "success.txt")) as file:
                    for line in file:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = line.split(": ")[-1]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(": ")[-1]

                        if line.startswith("higher_quantization_level"):
                            higher_quant_level = line.split(": ")[-1]

                        if line.startswith("compression"):
                            compression = line.split(": ")[-1]

                    if higher_quant_level:
                        label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
                    elif quant_level:
                        label = " ".join([reducer, quant_level, "bits"])
                    elif compression:
                        label = " ".join([reducer, "K:", compression])
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
            plt.figure(figsize=[10, 7])
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

            plt.grid()
            plt.xticks(events, GPUs)
            plt.ylabel("Images per sec")
            plt.title(f"Throughput Scalability {df_key} {instance}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./plots/throughput_scalability_{df_key}_{instance}.png")

            plt.show()


def plot_waiting_times(log_path):
    models = {"ResNet50": 1, "VGG16": 2}
    instances = ["P3 Waiting Time"]  # , "P3 Waiting Time Multi Node"]

    for instance in instances:
        Hs = os.listdir(os.path.join(log_path, instance))

        for H in Hs:
            GPUs = os.listdir(os.path.join(log_path, instance, H))
            GPUs.sort()

            [plt.figure(ind) for _, ind in models.items()]
            for GPU in GPUs:
                files = glob.glob(f"{log_path}/{instance}/{H}/{GPU}/*.pkl")

                for file in files:
                    model_name = file.split("_")[-1].split(".")[0]
                    plt.figure(models[model_name])

                    with open(file, "rb") as file:
                        waiting_time = pickle.load(file)

                    from scipy.stats import gaussian_kde

                    data = waiting_time[1:]
                    density = gaussian_kde(data)

                    if "Multi Node" in instance:
                        xs = np.linspace(0, 2e-4, 200)
                    else:
                        xs = np.linspace(0, 5e-5, 200)

                    # density.covariance_factor = lambda: .25
                    # density._compute_covariance()
                    plt.plot(xs, density(xs), label=GPU)
                    plt.title(f"{model_name}_{instance}_{H}")

                    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                    plt.legend()
                    plt.xlabel("Waiting Time")
                    plt.ylabel("Probability")
                    plt.grid()
                    plt.savefig(f"./plots/waiting_times_{model_name}_{H}_{instance}.png")
                    plt.grid()

            plt.show()


def plot_waiting_times_AWS(log_path):
    models = {"ResNet50": 1, "VGG16": 2}
    instances = ["P3 Waiting Time"]  # , "P3 Waiting Time Multi Node"]

    for instance in instances:
        Hs = os.listdir(os.path.join(log_path, instance))
        Hs.sort()

        for reducer in [
            "NoneAllReducer",
            "QSGDMaxNormReducer",
            "GlobalRandKMaxNormReducer",
            "QSGDMaxNormTwoScaleReducer",
            "GlobalRandKMaxNormTwoScaleReducer",
        ]:
            for H in Hs:
                GPUs = os.listdir(os.path.join(log_path, instance, H))
                GPUs.sort()

                [plt.figure(ind) for _, ind in models.items()]
                for GPU in GPUs:
                    experiments = glob.glob(f"{log_path}/{instance}/{H}/{GPU}/*")
                    experiments.sort()

                    for experiment in experiments:
                        with open(f"{experiment}/success.txt", "r") as success_file:
                            for line in success_file:
                                if line.startswith("reducer"):
                                    compressor = line.split(":")[-1].strip()

                                    if compressor == reducer:
                                        model_name = experiment.split("_")[-1].split(".")[0]
                                        plt.figure(models[model_name])

                                        files = glob.glob(f"{experiment}/*.pkl")
                                        files.sort()

                                        for file in files:
                                            worker_num = file.split("_")[-1].split(".")[0]

                                            with open(file, "rb") as file:
                                                waiting_time = pickle.load(file)

                                            from scipy.stats import gaussian_kde

                                            data = waiting_time[1:]
                                            density = gaussian_kde(data)

                                            if "Multi Node" in instance:
                                                xs = np.linspace(0, 2e-4, 200)
                                            else:
                                                xs = np.linspace(0, 5e-5, 200)

                                            # density.covariance_factor = lambda: .25
                                            # density._compute_covariance()
                                            plt.plot(xs, density(xs), label=f"{compressor} - {worker_num}")
                                            plt.title(f"{model_name}_{instance}_{GPU}_{H}")

                                            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                                            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                                            plt.legend()
                                            plt.xlabel("Waiting Time")
                                            plt.ylabel("Probability")
                                            plt.savefig(
                                                f"./plots/waiting_times_{model_name}_{H}_{instance}_{GPU}_{reducer}.png"
                                            )
                                    plt.show()


def plot_mean_variance_AWS(log_path, num_workers):
    models = {"ResNet50": 1, "VGG16": 2}
    instances = ["P3 Waiting Time"]  # , "P3 Waiting Time Multi Node"]

    for instance in instances:
        Hs = os.listdir(os.path.join(log_path, instance))
        Hs.sort()

        for reducer in [
            "NoneAllReducer",
            "QSGDMaxNormReducer",
            "GlobalRandKMaxNormReducer",
            "QSGDMaxNormTwoScaleReducer",
            "GlobalRandKMaxNormTwoScaleReducer",
        ]:
            mean = {model: [] for model in models}
            variance = {model: [] for model in models}

            mean_MC = {model: [] for model in models}
            variance_MC = {model: [] for model in models}

            steps = []
            [plt.figure(ind) for _, ind in models.items()]
            for H in Hs:
                steps.append(int(H.split("_")[-1]))
                GPUs = os.listdir(os.path.join(log_path, instance, H))
                GPUs.sort()

                [plt.figure(ind) for _, ind in models.items()]
                for GPU in GPUs:
                    if GPU != f"{num_workers} GPU":
                        print(f"Skip {GPU}")
                        continue

                    experiments = glob.glob(f"{log_path}/{instance}/{H}/{GPU}/*")
                    experiments.sort()

                    for experiment in experiments:
                        with open(f"{experiment}/success.txt", "r") as success_file:
                            for line in success_file:
                                if line.startswith("reducer"):
                                    compressor = line.split(":")[-1].strip()

                                    if compressor == reducer:
                                        model_name = experiment.split("_")[-1].split(".")[0]
                                        plt.figure(models[model_name])

                                        files = glob.glob(f"{experiment}/*.pkl")
                                        files.sort()

                                        worker_mean = []
                                        worker_variance = []

                                        for file in files:
                                            worker_num = file.split("_")[-1].split(".")[0]

                                            with open(file, "rb") as file:
                                                waiting_time = pickle.load(file)

                                            from scipy.stats import gaussian_kde

                                            data = waiting_time[1:]
                                            worker_mean.append(np.mean(data))
                                            worker_variance.append(np.var(data))

                                            n_samples = 1000000
                                            density = gaussian_kde(data)
                                            samples = density.resample(n_samples)

                                            worker_mean_mc = samples.mean()
                                            worker_variance_mc = samples.var()

                                        mean[model_name].append(np.mean(worker_mean))
                                        variance[model_name].append(np.mean(worker_variance))

                                        mean_MC[model_name].append(np.mean(worker_mean_mc))
                                        variance_MC[model_name].append(np.mean(worker_variance_mc))

            for model in models:
                plt.figure()
                plt.plot(steps, mean[model], label="Empirical")
                # plt.plot(steps, mean_MC[model], label='Monte Carlo')
                plt.title(f"Mean_WT_{model}_{reducer}")
                plt.legend()
                plt.ylabel("Waiting Time")
                plt.xlabel("H: Local steps")
                plt.savefig(f"./plots/mean_waiting_time_{model}_{num_workers} GPU_{instance}_{reducer}.png")
                plt.show()

                plt.figure()
                plt.plot(steps, variance[model], label="Empirical")
                # plt.plot(steps, variance_MC[model], label='Monte Carlo')
                plt.title(f"Variance_WT_{model}_{reducer}")
                plt.legend()
                plt.ylabel("Waiting Time")
                plt.xlabel("H: Local steps")
                plt.savefig(f"./plots/var_waiting_times_{model}_{num_workers} GPU_{instance}_{reducer}.png")
                plt.show()


def plot_reduce_times_AWS(log_path):
    models = {"ResNet50": 1, "VGG16": 2}
    instances = ["P3 Waiting Time"]  # , "P3 Waiting Time Multi Node"]

    for instance in instances:
        Hs = os.listdir(os.path.join(log_path, instance))
        Hs.sort()

        for reducer in [
            "NoneAllReducer",
            "QSGDMaxNormReducer",
            "GlobalRandKMaxNormReducer",
            "QSGDMaxNormTwoScaleReducer",
            "GlobalRandKMaxNormTwoScaleReducer",
        ]:
            for H in Hs:
                GPUs = os.listdir(os.path.join(log_path, instance, H))
                GPUs.sort()

                [plt.figure(ind) for _, ind in models.items()]
                for GPU in GPUs:
                    experiments = glob.glob(f"{log_path}/{instance}/{H}/{GPU}/*")
                    experiments.sort()

                    for experiment in experiments:
                        with open(f"{experiment}/success.txt", "r") as success_file:
                            for line in success_file:
                                if line.startswith("reducer"):
                                    compressor = line.split(":")[-1].strip()

                                    if compressor == reducer:
                                        model_name = experiment.split("_")[-1].split(".")[0]
                                        plt.figure(models[model_name])

                                        files = glob.glob(f"{experiment}/*.json")
                                        files.sort()

                                        for file in files:
                                            worker_num = file.split("_")[-1].split(".")[0]

                                            with open(file) as jsonfile:
                                                json_data = json.load(jsonfile)

                                            waiting_time = json_data["reduce_times"]

                                            from scipy.stats import gaussian_kde

                                            data = waiting_time[1:]
                                            density = gaussian_kde(data)

                                            if "Multi Node" in instance:
                                                xs = np.linspace(0, 2e-4, 200)
                                            else:
                                                xs = np.linspace(0, 2e-2 * (1 + int(H.split("_")[-1]) // 5), 200)

                                            # density.covariance_factor = lambda: .25
                                            # density._compute_covariance()
                                            plt.plot(xs, density(xs), label=f"{compressor} - {worker_num}")
                                            plt.title(f"{model_name}_{instance}_{GPU}_{H}")

                                            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                                            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                                            plt.legend()
                                            plt.xlabel("Reduce Time")
                                            plt.ylabel("Probability")
                                            plt.savefig(
                                                f"./plots/reduce_times_{model_name}_{H}_{instance}_{GPU}_{reducer}.png"
                                            )
                                    plt.show()


def plot_heterogenous_AWS(log_path):
    models = {"ResNet50": 1, "VGG16": 2}
    dynamic_batches = [
        "NAR",
        "NAR_128+8_128",
    ]  # , "NAR_128+16_128", "NAR_128+32_128", "NAR_128+64_128", "NAR_128+128_128"]

    for dynamic_batch in dynamic_batches:
        instances = os.listdir(os.path.join(log_path, dynamic_batch))
        instances.sort()

        for reducer in [
            "NoneAllReducer",
            "QSGDMaxNormReducer",
            "GlobalRandKMaxNormReducer",
            "QSGDMaxNormTwoScaleReducer",
            "GlobalRandKMaxNormTwoScaleReducer",
        ]:
            experiments_P2 = glob.glob(f"{log_path}/{dynamic_batch}/P2/*")
            experiments_P2.sort()
            experiments_P3 = glob.glob(f"{log_path}/{dynamic_batch}/P3/*")
            experiments_P3.sort()

            for experiment_P2, experiment_P3 in zip(experiments_P2, experiments_P3):
                with open(f"{experiment_P2}/success.txt", "r") as success_file:
                    for line in success_file:
                        if line.startswith("reducer"):
                            compressor = line.split(":")[-1].strip()

                            if compressor == reducer:
                                model_name = experiment_P3.split("_")[-1].split(".")[0]

                                plt.figure(models[model_name])

                                files = glob.glob(f"{log_path}/{dynamic_batch}/*/*{model_name}/*.json")
                                files.sort()

                                for file in files:
                                    worker_type = file.split("/")[5]

                                    with open(file) as jsonfile:
                                        json_data = json.load(jsonfile)

                                    batch_avg_time = json_data["batch"]["average_duration"]
                                    waiting_time = json_data["reduce_times"]

                                    from scipy.stats import gaussian_kde

                                    data = waiting_time[1:]
                                    density = gaussian_kde(data)

                                    xs = np.linspace(0, 1, 200)
                                    # density.covariance_factor = lambda: .25
                                    # density._compute_covariance()
                                    plt.plot(xs, density(xs), label=f"{compressor} - {worker_type}")
                                    plt.title(f"{model_name}_{dynamic_batch}")

                                    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                                    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                                    from matplotlib.ticker import FuncFormatter

                                    plt.gca().get_xaxis().set_major_formatter(
                                        FuncFormatter(lambda x, p: format(int(x / batch_avg_time * 100), ","))
                                    )

                                    plt.legend()
                                    plt.xlabel("Reduce Time")
                                    plt.ylabel("Probability")
                                    plt.savefig(f"./plots/reduce_times_{model_name}_{dynamic_batch}_{reducer}.png")
                        plt.show()


def plot_histogram_heterogenous_AWS(log_path):
    models = {"ResNet50": 1, "VGG16": 2}
    dynamic_batches = ["NAR", "NAR_128+8_128", "NAR_128+16_128", "NAR_128+32_128", "NAR_128+64_128", "NAR_128+128_128"]

    for dynamic_batch in dynamic_batches:
        instances = os.listdir(os.path.join(log_path, dynamic_batch))
        instances.sort()

        for reducer in [
            "NoneAllReducer",
            "QSGDMaxNormReducer",
            "GlobalRandKMaxNormReducer",
            "QSGDMaxNormTwoScaleReducer",
            "GlobalRandKMaxNormTwoScaleReducer",
        ]:
            experiments_P2 = glob.glob(f"{log_path}/{dynamic_batch}/P2/*")
            experiments_P2.sort()
            experiments_P3 = glob.glob(f"{log_path}/{dynamic_batch}/P3/*")
            experiments_P3.sort()

            for experiment_P2, experiment_P3 in zip(experiments_P2, experiments_P3):
                with open(f"{experiment_P2}/success.txt", "r") as success_file:
                    for line in success_file:
                        if line.startswith("reducer"):
                            compressor = line.split(":")[-1].strip()

                            if compressor == reducer:
                                model_name = experiment_P3.split("_")[-1].split(".")[0]

                                plt.figure(models[model_name])

                                files = glob.glob(f"{log_path}/{dynamic_batch}/*/*{model_name}/*.json")
                                files.sort()

                                for file in files:
                                    worker_type = file.split("/")[5]

                                    with open(file) as jsonfile:
                                        json_data = json.load(jsonfile)

                                    waiting_time = json_data["reduce_times"]

                                    from scipy.stats import gaussian_kde

                                    data = waiting_time[1:]
                                    density = gaussian_kde(data)

                                    xs = np.linspace(0, 1, 200)
                                    # density.covariance_factor = lambda: .25
                                    # density._compute_covariance()
                                    plt.hist(data, 50, label=f"{compressor} - {worker_type}")
                                    plt.title(f"{model_name}_{dynamic_batch}")

                                    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                                    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                                    plt.legend()
                                    plt.xlabel("Reduce Time")
                                    plt.ylabel("Probability")
                                    plt.savefig(
                                        f"./plots/reduce_times_histogram_{model_name}_{dynamic_batch}_{reducer}.png"
                                    )
                        plt.show()


def plot_mean_variance_reduce_time_AWS(log_path, num_workers):
    models = {"ResNet50": 1, "VGG16": 2}
    instances = ["P3 Waiting Time"]  # , "P3 Waiting Time Multi Node"]

    for instance in instances:
        Hs = os.listdir(os.path.join(log_path, instance))
        Hs.sort()

        for reducer in [
            "NoneAllReducer",
            "QSGDMaxNormReducer",
            "GlobalRandKMaxNormReducer",
            "QSGDMaxNormTwoScaleReducer",
            "GlobalRandKMaxNormTwoScaleReducer",
        ]:
            mean = {model: {} for worker in range(num_workers) for model in models}
            variance = {model: {} for worker in range(num_workers) for model in models}

            mean_MC = {model: [] for model in models}
            variance_MC = {model: [] for model in models}

            steps = []
            [plt.figure(ind) for _, ind in models.items()]
            for H in Hs:
                steps.append(int(H.split("_")[-1]))
                GPUs = os.listdir(os.path.join(log_path, instance, H))
                GPUs.sort()

                [plt.figure(ind) for _, ind in models.items()]
                for GPU in GPUs:
                    if GPU != f"{num_workers} GPU":
                        print(f"Skip {GPU}")
                        continue

                    experiments = glob.glob(f"{log_path}/{instance}/{H}/{GPU}/*")
                    experiments.sort()

                    for experiment in experiments:
                        with open(f"{experiment}/success.txt", "r") as success_file:
                            for line in success_file:
                                if line.startswith("reducer"):
                                    compressor = line.split(":")[-1].strip()

                                    if compressor == reducer:
                                        model_name = experiment.split("_")[-1].split(".")[0]
                                        plt.figure(models[model_name])

                                        files = glob.glob(f"{experiment}/*.json")
                                        files.sort()

                                        worker_mean = []
                                        worker_variance = []

                                        plt.figure(models[model_name])

                                        for file in files:
                                            worker_num = int(file.split("_")[-1].split(".")[0])

                                            with open(file) as jsonfile:
                                                json_data = json.load(jsonfile)

                                            waiting_time = json_data["reduce_times"]

                                            data = waiting_time[1:]
                                            worker_mean.append(np.mean(data))
                                            worker_variance.append(np.var(data))

                                            from scipy.stats import gaussian_kde

                                            n_samples = 1000000
                                            density = gaussian_kde(data)
                                            samples = density.resample(n_samples)

                                            worker_mean_mc = samples.mean()
                                            worker_variance_mc = samples.var()

                                            if not mean[model_name].get(worker_num):
                                                mean[model_name][worker_num] = [np.mean(data)]
                                                variance[model_name][worker_num] = [np.var(data)]
                                            else:
                                                mean[model_name][worker_num].append(np.mean(data))
                                                variance[model_name][worker_num].append(np.var(data))

                                        mean_MC[model_name].append(np.mean(worker_mean_mc))
                                        variance_MC[model_name].append(np.mean(worker_variance_mc))

            for model in models:
                plt.figure()
                for worker in range(num_workers):
                    plt.plot(steps, mean[model][worker], label=worker)

                # plt.plot(steps, mean_MC[model], label='Monte Carlo')
                plt.title(f"Mean_WT_{model}_{reducer}")
                plt.legend()
                plt.ylabel("Reduce Time")
                plt.xlabel("H: Local steps")
                plt.savefig(f"./plots/mean_reduce_times_{model}_{num_workers} GPU_{instance}_{reducer}.png")
                plt.show()

                plt.figure()
                for worker in range(num_workers):
                    plt.plot(steps, np.sqrt(variance[model][worker]), label=worker)

                # plt.plot(steps, variance_MC[model], label='Monte Carlo')
                plt.title(f"Variance_WT_{model}_{reducer}")
                plt.legend()
                plt.ylabel("Reduce Time")
                plt.xlabel("H: Local steps")
                plt.savefig(f"./plots/var_reduce_times_{model}_{num_workers} GPU_{instance}_{reducer}.png")
                plt.show()


def plot_performance_modelling(log_path):
    models = ["ResNet50", "VGG16"]
    instances = ["P3"]  # , "P3 Multi Node"]

    batch_size = 128
    inter_gpu_bw = 200 * 1024
    gpu_cpu_bw = 5 * 1024
    network_latency = 2.5e-3
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

                with open(os.path.join(experiment, "success.txt")) as file:
                    for line in file:
                        line = line.rstrip()

                        if line.startswith("architecture"):
                            architecture = line.split(": ")[-1]

                        if line.startswith("reducer"):
                            reducer = line.split(": ")[-1]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(": ")[-1]

                        if line.startswith("higher_quantization_level"):
                            higher_quant_level = line.split(": ")[-1]

                        if line.startswith("compression"):
                            compression = line.split(": ")[-1]

                    if higher_quant_level:
                        label = " ".join([reducer, quant_level, "&", higher_quant_level, "bits"])
                    elif quant_level:
                        label = " ".join([reducer, quant_level, "bits"])
                    elif compression:
                        label = " ".join([reducer, "K:", compression])
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

                if "NoneAllReducer" in label:
                    gradient_size = gradient_size
                elif "QSGDMaxNormReducer" in label:
                    gradient_size /= 4
                elif "QSGDMaxNormTwoScaleReducer" in label:
                    gradient_size /= 2
                elif "GlobalRandKMaxNormReducer" in label:
                    gradient_size *= 1 / 2000
                    gradient_size /= 4
                elif "GlobalRandKMaxNormTwoScaleReducer" in label:
                    gradient_size *= 1 / 2000
                    gradient_size /= 2
                else:
                    print(label)
                    raise ValueError("Method undefined")

                for gpu in GPUs:
                    if gpu > num_gpu_per_node:
                        num_nodes = gpu / num_gpu_per_node
                    else:
                        num_nodes = 1

                    # print(num_nodes, num_gpu_per_node)
                    # print(
                    #     time_df["batch"]["average_duration"],
                    #     gradient_size / inter_gpu_bw * np.log2(num_gpu_per_node),
                    #     (network_latency + gradient_size / network_bw) * np.log2(num_nodes),
                    # )

                    T = (
                        time_df["batch"]["average_duration"]
                        + gradient_size / inter_gpu_bw * np.log2(num_gpu_per_node)
                        + (network_latency + gradient_size / network_bw) * np.log2(num_nodes)
                    )
                    throughput = (batch_size * gpu) / T

                    throughput_results[compressor_ind_map[label]].append(int(throughput))

            throughput_dfs[models[group_ind]] = pd.DataFrame(throughput_results, index=compressor_ind_map.keys())

        for df_key in throughput_dfs:
            plt.figure(figsize=[10, 7])
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

            plt.grid()
            plt.xticks(events, GPUs)
            plt.ylabel("Images per sec")
            plt.title(f"Performance Modelling {df_key} {instance}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./plots/performance_modelling_{df_key}_{instance}.png")

            plt.show()


if __name__ == "__main__":
    root_log_path = "./logs/plot_logs/"

    plot_loss_curves(os.path.join(root_log_path, "convergence"))
    plot_loss_time_curves(os.path.join(root_log_path, "convergence"))
    # plot_top1_accuracy_curves(os.path.join(root_log_path, "convergence"))
    # plot_top1_accuracy_time_curves(os.path.join(root_log_path, "convergence"))
    # plot_top5_accuracy_curves(os.path.join(root_log_path, "convergence"))
    # plot_time_per_batch_curves(os.path.join(root_log_path, "convergence"))
    # plot_time_breakdown(os.path.join(root_log_path, "time_breakdown"))
    # plot_time_scalability(os.path.join(root_log_path, 'scalability'))
    # plot_throughput_scalability(os.path.join(root_log_path, 'scalability'))
    # plot_waiting_times(os.path.join(root_log_path, 'waiting_times'))
    # plot_waiting_times_AWS(os.path.join(root_log_path, 'waiting_times'))
    # plot_mean_variance_AWS(os.path.join(root_log_path, 'waiting_times'), 4)
    # plot_reduce_times_AWS(os.path.join(root_log_path, 'waiting_times'))
    # plot_heterogenous_AWS(os.path.join(root_log_path, "heterogenous"))
    # plot_histogram_heterogenous_AWS(os.path.join(root_log_path, 'heterogenous'))
    # plot_mean_variance_reduce_time_AWS(os.path.join(root_log_path, 'waiting_times'), 4)
    # plot_performance_modelling(os.path.join(root_log_path, "scalability"))

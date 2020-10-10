import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_curves(log_path):
    plt.figure(figsize=[10, 7])

    experiments = os.listdir(log_path)
    experiments.sort()

    for experiment in experiments:
        reducer = None
        quant_level = None
        compression = None

        with open(os.path.join(log_path, experiment, 'success.txt')) as file:
            for line in file:
                line = line.rstrip()
                if line.startswith("reducer"):
                    reducer = line.split(': ')[-1]

                if line.startswith("quantization_level"):
                    quant_level = line.split(': ')[-1]

                if line.startswith("compression"):
                    compression = line.split(': ')[-1]

        if quant_level:
            label = ' '.join([reducer, quant_level, 'bits'])
        elif compression:
            label = ' '.join([reducer, 'K:', compression])
        else:
            label = reducer

        log_dict = np.load(os.path.join(log_path, experiment, 'log_dict.npy'), allow_pickle=True)
        loss = log_dict[()].get('loss')
        plt.plot(loss, label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.legend()
    plt.savefig("./plots/loss.png")
    plt.show()


def plot_top1_accuracy_curves(log_path):
    plt.figure(figsize=[10, 7])

    experiments = os.listdir(log_path)
    experiments.sort()

    for experiment in experiments:
        reducer = None
        quant_level = None
        compression = None

        with open(os.path.join(log_path, experiment, 'success.txt')) as file:
            for line in file:
                line = line.rstrip()
                if line.startswith("reducer"):
                    reducer = line.split(': ')[-1]

                if line.startswith("quantization_level"):
                    quant_level = line.split(': ')[-1]

                if line.startswith("compression"):
                    compression = line.split(': ')[-1]

        if quant_level:
            label = ' '.join([reducer, quant_level, 'bits'])
        elif compression:
            label = ' '.join([reducer, 'K:', compression])
        else:
            label = reducer

        log_dict = np.load(os.path.join(log_path, experiment, 'log_dict.npy'), allow_pickle=True)
        top1_accuracy = log_dict[()].get('test_top1_accuracy')
        plt.plot(top1_accuracy, label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Top 1 Accuracy")
    plt.title("Accuracy curve")
    plt.legend()
    plt.savefig("./plots/top1.png")
    plt.show()


def plot_top5_accuracy_curves(log_path):
    plt.figure(figsize=[10, 7])

    experiments = os.listdir(log_path)
    experiments.sort()

    for experiment in experiments:
        reducer = None
        quant_level = None
        compression = None

        with open(os.path.join(log_path, experiment, 'success.txt')) as file:
            for line in file:
                line = line.rstrip()
                if line.startswith("reducer"):
                    reducer = line.split(': ')[-1]

                if line.startswith("quantization_level"):
                    quant_level = line.split(': ')[-1]

                if line.startswith("compression"):
                    compression = line.split(': ')[-1]

        if quant_level:
            label = ' '.join([reducer, quant_level, 'bits'])
        elif compression:
            label = ' '.join([reducer, 'K:', compression])
        else:
            label = reducer

        log_dict = np.load(os.path.join(log_path, experiment, 'log_dict.npy'), allow_pickle=True)
        top5_accuracy = log_dict[()].get('test_top5_accuracy')
        plt.plot(top5_accuracy, label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Top 5 Accuracy")
    plt.title("Accuracy curve")
    plt.legend()
    plt.savefig("./plots/top5.png")
    plt.show()


def plot_time_per_batch_curves(log_path):
    plt.figure(figsize=[10, 7])

    experiments = os.listdir(log_path)
    experiments.sort()

    for experiment in experiments:
        reducer = None
        quant_level = None
        compression = None

        with open(os.path.join(log_path, experiment, 'success.txt')) as file:
            for line in file:
                line = line.rstrip()
                if line.startswith("reducer"):
                    reducer = line.split(': ')[-1]

                if line.startswith("quantization_level"):
                    quant_level = line.split(': ')[-1]

                if line.startswith("compression"):
                    compression = line.split(': ')[-1]

            if quant_level:
                label = ' '.join([reducer, quant_level, 'bits'])
            elif compression:
                label = ' '.join([reducer, 'K:', compression])
            else:
                label = reducer

        log_dict = np.load(os.path.join(log_path, experiment, 'log_dict.npy'), allow_pickle=True)
        time = log_dict[()].get('time')
        epoch_time = np.zeros(len(time) - 1)

        for ind in range(epoch_time.shape[0]):
            epoch_time[ind] = time[ind+1] - time[ind]

        plt.plot(epoch_time, label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Average time")
    plt.title("Average time curve")
    plt.legend()
    plt.savefig("./plots/time.png")
    plt.show()


def plot_time_breakdown(log_path):
    time_labels = ['batch', 'batch.accumulate', 'batch.backward', 'batch.evaluate',
                   'batch.forward', 'batch.reduce', 'batch.step']

    models = ['ResNet50', 'VGG16']

    [plt.figure(num=ind, figsize=[10, 7]) for ind in range(len(models))]
    experiment_groups = [glob.glob(f'{log_path}/*{model}') for model in models]

    events = np.arange(len(time_labels))
    width = 0.15

    for group_ind, experiment_group in enumerate(experiment_groups):
        plt.figure(num=group_ind)
        experiment_group.sort()

        num_experiments = (len(experiment_group) - 1)

        for ind, experiment in enumerate(experiment_group):
            reducer = None
            quant_level = None
            compression = None

            with open(os.path.join(experiment, 'success.txt')) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("reducer"):
                        reducer = line.split(': ')[-1]

                    if line.startswith("quantization_level"):
                        quant_level = line.split(': ')[-1]

                    if line.startswith("compression"):
                        compression = line.split(': ')[-1]

                if quant_level:
                    label = ' '.join([reducer, quant_level, 'bits'])
                elif compression:
                    label = ' '.join([reducer, 'K:', compression])
                else:
                    label = reducer

            time_df = pd.read_json(os.path.join(experiment, 'timer_summary.json')).loc['average_duration']
            time_values = time_df[time_labels].values

            plt.bar(events + (ind - num_experiments / 2) * width, time_values, width, label=label)

        plt.xticks(events, time_labels)
        plt.ylabel("Average time")
        plt.title(f"Time breakdown {models[group_ind]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./plots/time_breakdown_{models[group_ind]}.png")

    plt.show()


def plot_time_scalability(log_path):
    time_labels = ['batch']
    # time_labels = ['batch.reduce']
    models = ['ResNet50', 'VGG16']

    [plt.figure(num=ind, figsize=[10, 7]) for ind in range(len(models))]

    GPUs = os.listdir(log_path)
    GPUs.sort()

    for GPU_ind, GPU in enumerate(GPUs):
        experiment_groups = [glob.glob(f'{log_path}/{GPU}/*{model}') for model in models]
        events = np.arange(len(GPUs))
        width = 0.1

        for group_ind, experiment_group in enumerate(experiment_groups):
            plt.figure(num=group_ind)
            experiment_group.sort()

            num_experiments = (len(experiment_group) - 1)

            for ind, experiment in enumerate(experiment_group):
                reducer = None
                quant_level = None
                compression = None

                with open(os.path.join(experiment, 'success.txt')) as file:
                    for line in file:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = line.split(': ')[-1]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(': ')[-1]

                        if line.startswith("compression"):
                            compression = line.split(': ')[-1]

                    if quant_level:
                        label = ' '.join([reducer, quant_level, 'bits'])
                    elif compression:
                        label = ' '.join([reducer, 'K:', compression])
                    else:
                        label = reducer

                time_df = pd.read_json(os.path.join(experiment, 'timer_summary.json')).loc['average_duration']
                num_iterations = pd.read_json(os.path.join(experiment, 'timer_summary.json')).loc['n_events']
                time_values = num_iterations * time_df[time_labels].values

                plt.bar(events[GPU_ind] + (ind - num_experiments / 2) * width, time_values, width, label=label)

            plt.xticks(events, GPUs)
            plt.ylabel("Time per epoch")
            plt.title(f"Time Scalability {models[group_ind]}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./plots/time_scalability_{models[group_ind]}.png")

    plt.show()


def plot_throughput_scalability(log_path):
    time_labels = ['batch']
    # time_labels = ['batch.reduce']
    models = ['ResNet50', 'VGG16']

    [plt.figure(num=ind, figsize=[10, 7]) for ind in range(len(models))]

    GPUs = os.listdir(log_path)
    GPUs.sort()

    for GPU_ind, GPU in enumerate(GPUs):
        experiment_groups = [glob.glob(f'{log_path}/{GPU}/*{model}') for model in models]
        events = np.arange(len(GPUs))
        width = 0.1

        for group_ind, experiment_group in enumerate(experiment_groups):
            plt.figure(num=group_ind)
            experiment_group.sort()

            num_experiments = (len(experiment_group) - 1)

            for ind, experiment in enumerate(experiment_group):
                reducer = None
                quant_level = None
                compression = None

                with open(os.path.join(experiment, 'success.txt')) as file:
                    for line in file:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = line.split(': ')[-1]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(': ')[-1]

                        if line.startswith("compression"):
                            compression = line.split(': ')[-1]

                    if quant_level:
                        label = ' '.join([reducer, quant_level, 'bits'])
                    elif compression:
                        label = ' '.join([reducer, 'K:', compression])
                    else:
                        label = reducer

                time_df = pd.read_json(os.path.join(experiment, 'timer_summary.json')).loc['average_duration']
                num_GPUs = int(GPUs[GPU_ind].split()[0])
                time_values = (128 * num_GPUs) / time_df[time_labels].values

                plt.bar(events[GPU_ind] + (ind - num_experiments / 2) * width, time_values, width, label=label)

            plt.xticks(events, GPUs)
            plt.ylabel("Images per sec")
            plt.title(f"Throughput Scalability {models[group_ind]}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./plots/throughput_scalability_{models[group_ind]}.png")

    plt.show()


if __name__ == '__main__':
    root_log_path = './logs/plot_logs/'

    # plot_loss_curves(os.path.join(root_log_path, 'convergence'))
    # plot_top1_accuracy_curves(os.path.join(root_log_path, 'convergence'))
    # plot_top5_accuracy_curves(os.path.join(root_log_path, 'convergence'))
    # plot_time_per_batch_curves(os.path.join(root_log_path, 'convergence'))
    # plot_time_breakdown(os.path.join(root_log_path, 'time_breakdown'))
    plot_time_scalability(os.path.join(root_log_path, 'scalability'))
    plot_throughput_scalability(os.path.join(root_log_path, 'scalability'))

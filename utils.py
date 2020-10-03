import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


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
    plt.figure(figsize=[10, 7])

    time_labels = ['batch', 'batch.accumulate', 'batch.backward', 'batch.evaluate',
                   'batch.forward', 'batch.reduce', 'batch.step', 'epoch_metrics.collect']

    experiments = os.listdir(log_path)
    experiments.sort()

    events = np.arange(len(time_labels))
    num_experiments = (len(experiments) - 1)
    width = 0.15

    for ind, experiment in enumerate(experiments):
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

        time_df = pd.read_json(os.path.join(log_path, experiment, 'timer_summary.json')).loc['average_duration']
        time_values = time_df[time_labels].values

        plt.bar(events + (ind - num_experiments / 2) * width, time_values, width, label=label)

    plt.xticks(events, time_labels)
    plt.xticks(rotation=90)
    plt.ylabel("Average time")
    plt.title("Time breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/time_breakdown.png")
    plt.show()


if __name__ == '__main__':
    plot_loss_curves('./logs/plot_logs')
    plot_top1_accuracy_curves('./logs/plot_logs')
    plot_top5_accuracy_curves('./logs/plot_logs')
    plot_time_per_batch_curves('./logs/plot_logs')
    plot_time_breakdown('./logs/plot_logs')

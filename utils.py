import os
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(log_path):
    plt.figure(figsize=[10, 7])

    experiments = os.listdir(log_path)
    experiments = ['2020_08_15_10_51_22_ResNet50', '2020_08_16_19_17_28_ResNet50', '2020_09_21_13_05_30_ResNet50', '2020_09_18_19_42_36_ResNet50']
    for experiment in experiments:
        reducer = None
        quant_level = None

        with open(os.path.join(log_path, experiment, 'success.txt')) as file:
            for line in file:
                line = line.rstrip()
                if line.startswith("reducer"):
                    reducer = line.split(': ')[-1]

                if line.startswith("quantization_level"):
                    quant_level = line.split(': ')[-1]

        if quant_level:
            label = ' '.join([reducer, quant_level, 'bits'])
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
    experiments = ['2020_08_15_10_51_22_ResNet50', '2020_08_16_19_17_28_ResNet50', '2020_09_21_13_05_30_ResNet50', '2020_09_18_19_42_36_ResNet50']
    for experiment in experiments:
        reducer = None
        quant_level = None

        with open(os.path.join(log_path, experiment, 'success.txt')) as file:
            for line in file:
                line = line.rstrip()
                if line.startswith("reducer"):
                    reducer = line.split(': ')[-1]

                if line.startswith("quantization_level"):
                    quant_level = line.split(': ')[-1]

        if quant_level:
            label = ' '.join([reducer, quant_level, 'bits'])
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
    experiments = ['2020_08_15_10_51_22_ResNet50', '2020_08_16_19_17_28_ResNet50', '2020_09_21_13_05_30_ResNet50', '2020_09_18_19_42_36_ResNet50']
    for experiment in experiments:
        reducer = None
        quant_level = None

        with open(os.path.join(log_path, experiment, 'success.txt')) as file:
            for line in file:
                line = line.rstrip()
                if line.startswith("reducer"):
                    reducer = line.split(': ')[-1]

                if line.startswith("quantization_level"):
                    quant_level = line.split(': ')[-1]

        if quant_level:
            label = ' '.join([reducer, quant_level, 'bits'])
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
    experiments = ['2020_08_15_10_51_22_ResNet50', '2020_08_16_19_17_28_ResNet50', '2020_09_21_13_05_30_ResNet50', '2020_09_18_19_42_36_ResNet50']
    for experiment in experiments:
        reducer = None
        quant_level = None

        with open(os.path.join(log_path, experiment, 'success.txt')) as file:
            for line in file:
                line = line.rstrip()
                if line.startswith("reducer"):
                    reducer = line.split(': ')[-1]

                if line.startswith("quantization_level"):
                    quant_level = line.split(': ')[-1]

        if quant_level:
            label = ' '.join([reducer, quant_level, 'bits'])
        else:
            label = reducer

        log_dict = np.load(os.path.join(log_path, experiment, 'log_dict.npy'), allow_pickle=True)
        time = log_dict[()].get('time')
        avg_time = np.zeros_like(time)

        for ind in range(avg_time.shape[0]):
            avg_time[ind] = (time[ind] - time[0]) / (ind + 1)

        plt.plot(avg_time, label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Average time")
    plt.title("Average time curve")
    plt.legend()
    plt.savefig("./plots/time.png")
    plt.show()


if __name__ == '__main__':
    plot_loss_curves('./logs')
    plot_top1_accuracy_curves('./logs')
    plot_top5_accuracy_curves('./logs')
    plot_time_per_batch_curves('./logs')

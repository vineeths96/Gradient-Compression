import os
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(log_path):
    plt.figure(figsize=[10, 7])

    for root, folder, files in os.walk(log_path):
        files.sort(reverse=True)
        reducer = None
        quant_level = None
        for file in files:
            if file.endswith('.json'):
                continue

            if file == 'success.txt':
                with open(os.path.join(root, file)) as infofile:
                    for line in infofile:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = line.split(': ')[-1]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(': ')[-1]

            if quant_level:
                label = ' '.join([reducer, quant_level, 'bits'])
            else:
                label = reducer

            if file.endswith('.npy'):
                log_dict = np.load(os.path.join(root, file), allow_pickle=True)
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

    for root, folder, files in os.walk(log_path):
        files.sort(reverse=True)
        reducer = None
        quant_level = None
        for file in files:
            if file.endswith('.json'):
                continue

            if file == 'success.txt':
                with open(os.path.join(root, file)) as infofile:
                    for line in infofile:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = line.split(': ')[-1]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(': ')[-1]

            if quant_level:
                label = ' '.join([reducer, quant_level, 'bits'])
            else:
                label = reducer

            if file.endswith('.npy'):
                log_dict = np.load(os.path.join(root, file), allow_pickle=True)
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

    for root, folder, files in os.walk(log_path):
        files.sort(reverse=True)
        reducer = None
        quant_level = None
        for file in files:
            if file.endswith('.json'):
                continue

            if file == 'success.txt':
                with open(os.path.join(root, file)) as infofile:
                    for line in infofile:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = line.split(': ')[-1]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(': ')[-1]

            if quant_level:
                label = ' '.join([reducer, quant_level, 'bits'])
            else:
                label = reducer

            if file.endswith('.npy'):
                log_dict = np.load(os.path.join(root, file), allow_pickle=True)
                top1_accuracy = log_dict[()].get('test_top5_accuracy')
                plt.plot(top1_accuracy, label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Top 5 Accuracy")
    plt.title("Accuracy curve")
    plt.legend()
    plt.savefig("./plots/top5.png")
    plt.show()


def plot_time_per_batch_curves(log_path):
    plt.figure(figsize=[10, 7])

    for root, folder, files in os.walk(log_path):
        files.sort(reverse=True)
        reducer = None
        quant_level = None
        for file in files:
            if file.endswith('.json'):
                continue

            if file == 'success.txt':
                with open(os.path.join(root, file)) as infofile:
                    for line in infofile:
                        line = line.rstrip()
                        if line.startswith("reducer"):
                            reducer = line.split(': ')[-1]

                        if line.startswith("quantization_level"):
                            quant_level = line.split(': ')[-1]

            if quant_level:
                label = ' '.join([reducer, quant_level, 'bits'])
            else:
                label = reducer

            if file.endswith('.npy'):
                log_dict = np.load(os.path.join(root, file), allow_pickle=True)
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

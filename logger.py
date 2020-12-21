import os
import torch
import datetime
import numpy as np


class Logger:
    """
    Logs information and model statistics
    """

    def __init__(self, log_path, config):
        os.makedirs(log_path)
        self._log_path = log_path
        self._config = config
        self._start = datetime.datetime.now()

        metric_list = {
            "train_top1_accuracy",
            "train_top5_accuracy",
            "test_top1_accuracy",
            "test_top5_accuracy",
            "loss",
            "time",
        }
        self._log_dict = {metric: np.zeros(self._config["num_epochs"]) for metric in metric_list}

    def log_info(self, name, values, tags={}):
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

    def epoch_update(self, epoch, epoch_metrics, test_stats):
        self._log_dict["train_top1_accuracy"][epoch] = epoch_metrics.values()["top1_accuracy"]
        self._log_dict["train_top5_accuracy"][epoch] = epoch_metrics.values()["top5_accuracy"]
        self._log_dict["test_top1_accuracy"][epoch] = test_stats.values()["top1_accuracy"]
        self._log_dict["test_top5_accuracy"][epoch] = test_stats.values()["top5_accuracy"]
        self._log_dict["loss"][epoch] = epoch_metrics.values()["cross_entropy_loss"]
        self._log_dict["time"][epoch] = (datetime.datetime.now() - self._start).total_seconds()

    def summary_writer(self, model, timer, bits_communicated):
        timer.save_summary(f"{self._log_path}/timer_summary.json")

        with open(f"{self._log_path}/success.txt", "w") as file:
            file.write(f"Training completed at {datetime.datetime.now()}\n\n")

            file.write(f"Training parameters\n")
            list_of_strings = [f"{key} : {value}" for key, value in self._config.items()]
            [file.write(f"{string}\n") for string in list_of_strings]

            file.write(f"Bits communicated: {bits_communicated}")

        np.save(f"{self._log_path}/log_dict.npy", self._log_dict)
        torch.save(model.state_dict(), f"{self._log_path}/model.pt")

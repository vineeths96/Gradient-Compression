import os
import torch
import datetime
import numpy as np


class Logger:
    """
    Logs information and model statistics
    """

    def __init__(self, config, local_rank):
        self._log_path = f"./logs/{self.get_log_path().strftime('%Y_%m_%d_%H_%M_%S')}_{config['architecture']}"

        if local_rank == 0:
            os.makedirs(self._log_path, exist_ok=True)

        self._local_rank = local_rank
        self._config = config
        self._start = datetime.datetime.now()

        metric_list = {
            "train_top1_accuracy",
            "train_top5_accuracy",
            "test_top1_accuracy",
            "test_top5_accuracy",
            "train_loss",
            "test_loss",
            "time",
        }
        self._log_dict = {metric: np.zeros([config["runs"], self._config["num_epochs"]]) for metric in metric_list}

    def get_log_path(self, dt=None, roundTo=30):
        if not dt:
            dt = datetime.datetime.now()

        seconds = (dt.replace(tzinfo=None) - dt.min).seconds
        rounding = (seconds + roundTo / 2) // roundTo * roundTo

        return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)

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

    def epoch_update(self, run, epoch, epoch_metrics, test_stats):
        self._log_dict["train_top1_accuracy"][run, epoch] = epoch_metrics.values()["top1_accuracy"]
        self._log_dict["train_top5_accuracy"][run, epoch] = epoch_metrics.values()["top5_accuracy"]
        self._log_dict["test_top1_accuracy"][run, epoch] = test_stats.values()["top1_accuracy"]
        self._log_dict["test_top5_accuracy"][run, epoch] = test_stats.values()["top5_accuracy"]
        self._log_dict["train_loss"][run, epoch] = epoch_metrics.values()["cross_entropy_loss"]
        self._log_dict["test_loss"][run, epoch] = test_stats.values()["cross_entropy_loss"]
        self._log_dict["time"][run, epoch] = (datetime.datetime.now() - self._start).total_seconds()

    def save_model(self, model):
        torch.save(model.state_dict(), f"{self._log_path}/model.pt")

    def summary_writer(self, timer, best_accuracy, bits_communicated):
        timer.save_summary(f"{self._log_path}/timer_summary_{self._local_rank}.json")

        if self._local_rank == 0:
            with open(f"{self._log_path}/success.txt", "w") as file:
                file.write(f"Training completed at {datetime.datetime.now()}\n\n")

                file.write(f"Best Top 1 Accuracy: {sum(best_accuracy['top1']) / len(best_accuracy['top1'])}\n")
                file.write(f"Best Top 5 Accuracy: {sum(best_accuracy['top5']) / len(best_accuracy['top5'])}\n\n")

                file.write(f"Training parameters\n")
                list_of_strings = [f"{key} : {value}" for key, value in self._config.items()]
                [file.write(f"{string}\n") for string in list_of_strings]

                file.write(f"Bits communicated: {bits_communicated}")

            np.save(f"{self._log_path}/log_dict.npy", self._log_dict)

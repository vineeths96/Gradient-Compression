import torch
import torch.distributed


class AverageMeter:
    """
    Averages the tracked values across all workers.
    """

    def __init__(self, device):
        self._device = device
        self._average = {}
        self._counter = 0

    def values(self):
        return self._average

    def add(self, metric_data):
        self._counter += 1

        for key, value in metric_data.items():
            self._average[key] = value

    def reduce(self):
        if not torch.distributed.is_available():
            return

        total_count = torch.tensor(self._counter, dtype=torch.int32, device=self._device)
        count_reduce_op = torch.distributed.all_reduce(tensor=total_count, async_op=True)
        count_reduce_op.wait()

        for key in self._average:
            tensor = torch.tensor(self._average[key], dtype=torch.float16, device=self._device)
            tensor.mul_(self._counter)
            reduce_op = torch.distributed.all_reduce(tensor=tensor, async_op=True)
            reduce_op.wait()

            self._average[key] = (tensor / total_count).item()

        self._counter = total_count.item()

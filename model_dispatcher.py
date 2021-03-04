import torch
import torchvision
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

import models
from metrics import AverageMeter


class CIFAR:
    """
    Instantiates a deep model of the specified architecture on the specified device.
    """

    def __init__(self, device, timer, architecture, seed):
        self._device = device
        self._timer = timer
        self._architecture = architecture
        self._seed = seed

        self._epoch = 0
        self._model = self._create_model()
        self._train_set, self._test_set = self._load_dataset()

        self.len_train_loader = None
        self.len_aux_train_loader = None
        self.len_test_loader = None

        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)
        self.parameters = [parameter for parameter in self._model.parameters()]

    def _load_dataset(self, data_path="./data"):
        mean = (0.4914, 0.4822, 0.4465)
        std_dev = (0.247, 0.243, 0.261)

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std_dev),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std_dev),
            ]
        )

        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

        return train_set, test_set

    def _create_model(self):
        torch.random.manual_seed(self._seed)
        model = getattr(models, self._architecture)()
        model.to(self._device)
        model.train()

        return model

    def train_dataloader(self, batch_size=32):
        train_sampler = DistributedSampler(dataset=self._train_set)
        train_sampler.set_epoch(self._epoch)

        train_loader = DataLoader(
            dataset=self._train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=dist.get_world_size(),
        )

        self.len_train_loader = len(train_loader)

        for imgs, labels in train_loader:
            imgs = imgs.to(self._device)
            labels = labels.to(self._device)

            yield imgs, labels

        self._epoch += 1

    def auxiliary_train_dataloader(self, batch_size=32):
        train_sampler = DistributedSampler(dataset=self._train_set)
        train_sampler.set_epoch(self._epoch)

        train_loader = DataLoader(
            dataset=self._train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=dist.get_world_size(),
        )

        self.len_aux_train_loader = len(train_loader)

        for imgs, labels in train_loader:
            imgs = imgs.to(self._device)
            labels = labels.to(self._device)

            yield imgs, labels

    def test_dataloader(self, batch_size=32):
        test_sampler = DistributedSampler(dataset=self._test_set)

        test_loader = DataLoader(
            dataset=self._test_set,
            batch_size=batch_size,
            sampler=test_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=dist.get_world_size(),
        )

        self.len_test_loader = len(test_loader)

        for imgs, labels in test_loader:
            imgs = imgs.to(self._device)
            labels = labels.to(self._device)

            yield imgs, labels

    def batch_loss(self, batch):
        with torch.no_grad():
            imgs, labels = batch

            with self._timer("batch.forward", float(self._epoch)):
                prediction = self._model(imgs)
                loss = self._criterion(prediction, labels)

            with self._timer("batch.evaluate", float(self._epoch)):
                metrics = self.evaluate_predictions(prediction, labels)

        return loss.item(), metrics

    def batch_loss_with_gradients(self, batch):
        self._model.zero_grad()
        imgs, labels = batch

        with self._timer("batch.forward", float(self._epoch)):
            prediction = self._model(imgs)
            loss = self._criterion(prediction, labels)

        with self._timer("batch.backward", float(self._epoch)):
            loss.backward()

        with self._timer("batch.evaluate", float(self._epoch)):
            metrics = self.evaluate_predictions(prediction, labels)

        grad_vec = [parameter.grad for parameter in self._model.parameters()]

        return loss.detach(), grad_vec, metrics

    def auxiliary_batch_loss_with_gradients(self, batch):
        imgs, labels = batch

        with self._timer("batch.auxiliary.forward", float(self._epoch)):
            prediction = self._model(imgs)
            loss = self._criterion(prediction, labels)

        with self._timer("batch.auxiliary.backward", float(self._epoch)):
            loss.backward()

        with self._timer("batch.auxiliary.evaluate", float(self._epoch)):
            metrics = self.evaluate_predictions(prediction, labels)

        grad_vec = [parameter.grad for parameter in self._model.parameters()]

        return loss.detach(), grad_vec, metrics

    def evaluate_predictions(self, pred_labels, true_labels):
        def accuracy(output, target, topk=(1,)):
            maxk = max(topk)
            batch_size = true_labels.size()[0]

            _, pred_topk = output.topk(maxk, 1, True, True)
            pred_topk = pred_topk.t()
            correct = pred_topk.eq(target.view(1, -1).expand_as(pred_topk))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1 / batch_size))

            return res

        with torch.no_grad():
            cross_entropy_loss = self._criterion(pred_labels, true_labels)
            top1_accuracy, top5_accuracy = accuracy(pred_labels, true_labels, topk=(1, 5))

        return {
            "cross_entropy_loss": cross_entropy_loss.item(),
            "top1_accuracy": top1_accuracy.item(),
            "top5_accuracy": top5_accuracy.item(),
        }

    def state_dict(self):
        return self._model.state_dict()

    def test(self, batch_size=256):
        test_loader = self.test_dataloader(batch_size=batch_size)

        mean_metrics = AverageMeter(self._device)
        test_model = self._model
        test_model.eval()

        for i, batch in enumerate(test_loader):
            # print("Test ", i/self.len_test_loader)
            with torch.no_grad():
                imgs, labels = batch
                prediction = test_model(imgs)
                metrics = self.evaluate_predictions(prediction, labels)

            mean_metrics.add(metrics)

        # print("Test Acc", mean_metrics.values()["top1_accuracy"])
        mean_metrics.reduce()
        test_model.train()

        return mean_metrics

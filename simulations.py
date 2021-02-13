import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# def plot_waiting_times(process_time):
#     waiting_time = torch.zeros_like(process_time)
#
#     for iteration in range(process_time.size(0)):
#         for worker in range(process_time.size(1)):
#             waiting_time[iteration, worker] = process_time[iteration, :].max() - process_time[iteration, worker] + torch.distributions.exponential.Exponential(torch.Tensor([5e5])).sample()
#
#     print(waiting_time)
#     data = waiting_time[:, 1]
#     density = gaussian_kde(data)
#
#     xs = np.linspace(0, 5e-5, 200)
#
#     # density.covariance_factor = lambda: .25
#     # density._compute_covariance()
#     plt.plot(xs, density(xs), label='GPU')
#     # plt.title(f'{model_name}_{instance}_{L}')
#
#     plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
#     plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#
#     plt.legend()
#     plt.grid()
#     # plt.savefig(f"./plots/waiting_times_{model_name}_{L}_{instance}.png")
#     plt.show()
#
#
# def simulate_waiting_times(num_workers, num_iterations):
#     X_distributions = [torch.distributions.exponential.Exponential(torch.Tensor([1e5]))] * num_workers
#     X_distributions = [torch.distributions.normal.Normal(torch.tensor([2.5e-5 + ind / 100000]), torch.tensor([5e-6 * ind / 100])) for ind in range(num_workers)]
#
#     X = torch.zeros([num_iterations, num_workers])
#
#     for iteration in range(num_iterations):
#         for worker in range(num_workers):
#             X[iteration, worker] = X_distributions[worker].sample()
#
#     print(X)
#     plot_waiting_times(X)


def plot_waiting_times(process_time):
    waiting_time = torch.zeros_like(process_time)

    for iteration in range(process_time.size(0)):
        com_time = torch.distributions.normal.Normal(torch.tensor([2e-5]), torch.tensor([1e-6])).sample()

        for worker in range(process_time.size(1)):
            waiting_time[iteration, worker] = (
                process_time[iteration, :].max() - process_time[iteration, worker]
            )  # + com_time

    print(waiting_time)

    for ind in range(4):
        data = waiting_time[:, ind]
        density = gaussian_kde(data)

        xs = np.linspace(0, 7e-5, 200)

        # density.covariance_factor = lambda: 0.25
        # density._compute_covariance()
        plt.plot(xs, density(xs), label=f"{ind}")

        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.legend()
    plt.grid()
    plt.xlabel("Waiting Time")
    plt.ylabel("Probability")
    # plt.savefig(f"./plots/waiting_times_{model_name}_{L}_{instance}.png")
    plt.show()


def simulate_waiting_times(num_workers, num_iterations):
    mean_distribution = torch.distributions.normal.Normal(torch.tensor([5e-5]), torch.tensor([1e-8]))
    # mean_distribution = torch.distributions.uniform.Uniform(torch.tensor([-5e-3]), torch.tensor([5e-3]))
    lambda_distribution = torch.distributions.uniform.Uniform(torch.tensor([-5e2]), torch.tensor([5e2]))

    X_distributions = [
        torch.distributions.exponential.Exponential(torch.Tensor([1e5 + lambda_distribution.sample()]))
        for ind in range(num_workers)
    ]
    # X_distributions = [torch.distributions.normal.Normal(torch.tensor([2e-4 + mean_distribution.sample()]), torch.tensor([1e-5])) for ind in range(num_workers)]
    # X_distributions = [torch.distributions.normal.Normal(torch.tensor([2e-4 + ind * mean_distribution.sample()]), torch.tensor([1e-5])) for ind in range(num_workers)]

    X = torch.zeros([num_iterations, num_workers])

    for iteration in range(num_iterations):
        for worker in range(num_workers):
            X[iteration, worker] = X_distributions[worker].sample().abs()

    print(X)
    plot_waiting_times(X)


simulate_waiting_times(4, 30)

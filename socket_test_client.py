import time
import torch
from socket_com import ClientTCP


# REPS = 10
# MSG_SIZES = [1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 25e6]
# client = ClientTCP(SERVER="10.217.16.13")
#
# for msg in MSG_SIZES:
#     time_sum = 0
#     for i in range(REPS):
#         start = time.time()
#         client.send(torch.arange(msg).to(torch.float32))
#         time_elapsed = time.time() - start
#         time_sum += time_elapsed
#         # print(time_elapsed)
#
#     # print("Total: ", time_sum)
#     print(f"MSG: [{msg}] Avg TIme: {time_sum / REPS}")

client = ClientTCP(SERVER="10.217.16.13")

start = time.time()
client.send(torch.arange(msg).to(torch.float32))
time_elapsed = time.time() - start
print(time_elapsed)

# print("Total: ", time_sum)
print(f"MSG: [{msg}] Avg TIme: {time_sum / REPS}")

client.send(torch.tensor(float('inf')))

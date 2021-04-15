import time
import torch
from socket_com import ClientTCP, ClientUDP


# use_TCP = True
use_TCP = False

if use_TCP:
    REPS = 1
    MSG_SIZES = [1000]
    # MSG_SIZES = [1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 25e6]
    client = ClientTCP(SERVER="10.216.18.179")

    for msg in MSG_SIZES:
        time_sum = 0
        for i in range(REPS):
            start = time.time()

            client.send(torch.cat([torch.arange(msg).unsqueeze(1), 2 * torch.arange(msg).unsqueeze(1)], dim=-1).to(torch.float32))
            # client.send(torch.arange(msg).to(torch.float32)#.to_sparse())
            time_elapsed = time.time() - start
            time_sum += time_elapsed
            # print(time_elapsed)

        print("Total: ", time_sum)
        print(f"MSG: [{msg}] Avg Time: {time_sum / REPS}")
else:
    REPS = 1
    MSG_SIZES = [1000]
    # MSG_SIZES = [1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 25e6]
    client = ClientUDP(SERVER="10.216.18.179")

    for msg in MSG_SIZES:
        time_sum = 0
        for i in range(REPS):
            CHUNK = 100
            messages = torch.cat([torch.arange(msg).unsqueeze(1), 1 * torch.arange(msg).unsqueeze(1)], dim=-1).to(torch.float32).split(CHUNK)

            start = time.time()
            for message in messages:
                client.send(message.clone())

                import time
                time.sleep(0.001)

            time_elapsed = time.time() - start
            time_sum += time_elapsed
            # print(time_elapsed)

        print("Total: ", time_sum)
        print(f"MSG: [{msg}] Avg Time: {time_sum / REPS}")

client.send(torch.tensor(float("inf")))

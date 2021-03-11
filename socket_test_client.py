# import logging
# import numpy as np
# from socket_com import SocketCom, SocketComTCP
#
# logger = logging.getLogger('simple client')
# logger.setLevel(logging.INFO)
#
# logger.info("waiting for frame")
#
# npSocket = SocketCom()
# # npSocket = SocketComTCP()
#
# npSocket.startClient(9999)
#
# for i in range(10):
#     frame = npSocket.recieveNumpy()
#
# logger.info("frame recieved")
# logger.info(frame)
#
# try:
#     npSocket.endServer()
# except OSError as err:
#     logging.error("server already disconnected")


import torch
from socket_com import Client

client = Client(SERVER="10.217.16.13")

import time
s = time.time()
client.send(torch.arange(23e3))
print(time.time() -s)

import time
time.sleep(4)

torch.tensor(float('inf'))

# import torch
# import logging
# import numpy as np
# from time import sleep
# from socket_com import SocketCom, SocketComTCP
#
# logger = logging.getLogger('Simple server')
# logger.setLevel(logging.INFO)
#
# host_ip = '10.217.16.13'
#
# npSocket = SocketCom()
# # npSocket = SocketComTCP()
#
# while True:
#     try:
#         npSocket.startServer(host_ip, 9999)
#         break
#     except:
#         logger.warning("Connection failed, trying again.")
#         sleep(1)
#         continue
#
# frame = torch.arange(2500).reshape(-1, 2).to(torch.int32)
# logger.info("Sending frame: ")
# logger.info(frame)
#
# # npSocket.sendNumpy(frame[:250] * 1)
#
# # for i in range(10):
# #     npSocket.sendNumpy(frame[i * 125: (i+1) * 125, :] * 1)
#
#     # import time
#     # time.sleep(0.1)
#
# try:
#     npSocket.endServer()
# except OSError as err:
#     logging.error("Client already disconnected")


from socket_com import Server

server = Server(SERVER="10.217.16.13")
server.start()
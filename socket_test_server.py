from socket_com import ServerTCP, ServerUDP

# server = ServerTCP(SERVER="10.216.18.179", MSG_SIZE=100000)
server = ServerUDP(SERVER="10.216.18.179", MSG_SIZE=10000)


server.start()

from socket_com import ServerTCP, ServerUDP

# server = ServerTCP(SERVER="10.217.16.13")
server = ServerUDP(SERVER="10.217.16.13")

server.start()

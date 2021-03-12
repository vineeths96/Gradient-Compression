import io
import torch
import socket
import threading


class ServerTCP:
    def __init__(self, HEADER=64, PORT=5050, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.SERVER = SERVER
        self.ADDR = (SERVER, PORT)
        self.FORMAT = "utf-8"
        self.DISCONNECT_MESSAGE = torch.tensor(float("inf"))

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(self.ADDR)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        packet_size = len(file.getvalue())
        header = "{0}:".format(packet_size)
        header = bytes(header.encode())

        encoded = bytearray()
        encoded += header

        file.seek(0)
        encoded += file.read()

        return encoded

    def decode(self, buffer):
        tensor = torch.load(io.BytesIO(buffer))

        return tensor

    def handle_client(self, conn, addr):
        print(f"[NEW CONNECTION] {addr} connected.")

        connected = True
        while connected:
            length = None
            buffer = bytearray()

            readnext = True
            while readnext:
                msg = conn.recv(1024)
                buffer += msg

                # print(len(buffer))
                if len(buffer) == length:
                    readnext = False

                if length is None:
                    if b":" not in buffer:
                        break

                    length_str, ignored, buffer = buffer.partition(b":")
                    length = int(length_str)

                    if len(buffer) == length:
                        readnext = False

            buffer = buffer[:length]

            msg = self.decode(buffer)
            print(f"[{addr}] {msg}")
            conn.send("Message received".encode(self.FORMAT))

            if not len(msg.shape) and torch.isinf(msg):
                connected = False
                print(f"[DROP CONNECTION] {addr} closed")
                conn.shutdown(1)
                conn.close()

    def start(self):
        self.server.listen()
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        while True:
            conn, addr = self.server.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()
            print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")

    def stop(self):
        self.server.shutdown(1)
        self.server.close()


class ClientTCP:
    def __init__(self, HEADER=64, PORT=5050, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.SERVER = SERVER
        self.ADDR = (SERVER, PORT)
        self.FORMAT = "utf-8"
        self.DISCONNECT_MESSAGE = torch.tensor(float("inf"))

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(self.ADDR)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        packet_size = len(file.getvalue())
        header = "{0}:".format(packet_size)
        header = bytes(header.encode())

        encoded = bytearray()
        encoded += header

        file.seek(0)
        encoded += file.read()

        return encoded

    def decode(self, buffer):
        tensor = torch.load(io.BytesIO(buffer))

        return tensor

    def send(self, tensor):
        message = self.encode(tensor)

        self.client.send(message)
        print(self.client.recv(2048).decode(self.FORMAT))


class ServerUDP:
    def __init__(self, HEADER=64, PORT=5050, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.SERVER = SERVER
        self.ADDR = (SERVER, PORT)
        self.FORMAT = "utf-8"
        self.DISCONNECT_MESSAGE = torch.tensor(float("inf"))

        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind(self.ADDR)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        packet_size = len(file.getvalue())
        header = "{0}:".format(packet_size)
        header = bytes(header.encode())

        encoded = bytearray()
        encoded += header

        file.seek(0)
        encoded += file.read()

        return encoded

    def decode(self, buffer):
        tensor = torch.load(io.BytesIO(buffer))

        return tensor

    def handle_client(self):
        print(f"[NEW CONNECTION] {0} connected.")

        connected = True
        while connected:
            length = None
            buffer = bytearray()

            readnext = True
            while readnext:
                msg, addr = self.server.recvfrom(1024*2)
                buffer += msg

                # print(len(buffer))
                if len(buffer) == length:
                    readnext = False

                if length is None:
                    if b":" not in buffer:
                        break

                    length_str, ignored, buffer = buffer.partition(b":")
                    length = int(length_str)

                    if len(buffer) == length:
                        readnext = False

            buffer = buffer[:length]

            msg = self.decode(buffer)
            print(f"[{addr}] {msg}")
            self.server.sendto("Message received".encode(self.FORMAT), addr)

            if not len(msg.shape) and torch.isinf(msg):
                connected = False
                print(f"[DROP CONNECTION] {addr} closed")

    def start(self):
        # self.server.listen()
        print(f"[LISTENING] Server is listening on {self.SERVER}")

        while True:
            # conn, addr = self.server.accept()
            # thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            # thread.start()
            # print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")

            self.handle_client()
            # data = self.server.recv(1024)
            # print(self.decode(data))
            # self.server.send("Message received".encode(self.FORMAT))

    def stop(self):
        self.server.shutdown(1)
        self.server.close()


class ClientUDP:
    def __init__(self, HEADER=64, PORT=5050, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.SERVER = SERVER
        self.ADDR = (SERVER, PORT)
        self.FORMAT = "utf-8"
        self.DISCONNECT_MESSAGE = torch.tensor(float("inf"))

        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client.connect(self.ADDR)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        packet_size = len(file.getvalue())
        header = "{0}:".format(packet_size)
        header = bytes(header.encode())

        encoded = bytearray()
        encoded += header

        file.seek(0)
        encoded += file.read()

        return encoded

    def decode(self, buffer):
        tensor = torch.load(io.BytesIO(buffer))

        return tensor

    def send(self, tensor):
        message = self.encode(tensor)

        self.client.send(message)
        # print(self.client.recv(2048).decode(self.FORMAT))

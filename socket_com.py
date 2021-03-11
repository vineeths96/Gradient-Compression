import io
import torch
import socket
import threading


class Server:
    def __init__(self, HEADER=64, PORT=5050, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.SERVER = SERVER
        self.ADDR = (SERVER, PORT)
        self.FORMAT = 'utf-8'
        self.DISCONNECT_MESSAGE = torch.tensor(float('inf'))

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind(self.ADDR)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        packet_size = len(file.getvalue())
        header = '{0}:'.format(packet_size)
        header = bytes(header.encode())  # prepend length of array

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

        length = None
        buffer = bytearray()
        connected = True
        while connected:
            msg = conn.recv(1024)
            buffer += msg

            if len(buffer) == length:
                break

            while True:
                if length is None:
                    if b':' not in buffer:
                        break

                    length_str, ignored, buffer = buffer.partition(b':')
                    length = int(length_str)

                if len(buffer) < length:
                    break

                buffer = buffer[:length]
                length = None
                break

        msg = self.decode(buffer)

            # if not len(msg.shape) and torch.isinf(msg):
            #     print(msg)
            #     connected = False

        print(f"[{addr}] {msg}")
        conn.send("Message received".encode(self.FORMAT))

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


class Client:
    def __init__(self, HEADER=64, PORT=5050, SERVER=socket.gethostbyname(socket.gethostname())):
        self.HEADER = HEADER
        self.PORT = PORT
        self.SERVER = SERVER
        self.ADDR = (SERVER, PORT)
        self.FORMAT = 'utf-8'
        self.DISCONNECT_MESSAGE = "!DISCONNECT"

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client.connect(self.ADDR)

    def encode(self, tensor):
        file = io.BytesIO()
        torch.save(tensor, file)

        packet_size = len(file.getvalue())
        header = '{0}:'.format(packet_size)
        header = bytes(header.encode())  # prepend length of array

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

        # print(message)

        # msg_length = len(message)
        # send_length = str(msg_length).encode(self.FORMAT)
        # send_length += b' ' * (self.HEADER - len(send_length))
        # self.client.send(send_length)
        # print(self.decode(message))

        self.client.send(message)
        print(self.client.recv(2048).decode(self.FORMAT))

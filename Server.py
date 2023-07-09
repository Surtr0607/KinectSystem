import pickle
import socket
import threading

import numpy
from socket import *


class Server:
    def __init__(self):
        self.conn_pool = []  # connection pool
        # bind the socket to the address and port number
        self.socket_server = socket(AF_INET, SOCK_STREAM)
        host = 'localhost'  # obtain local address
        port = 2456  # set port number
        self.socket_server.bind((host, port))
        self.socket_server.listen(2)

    def accept_client(self):
        while True:
            client, addr = self.socket_server.accept()  # connect establish
            print("Connection " + str(addr[0]) + " established")
            self.conn_pool.append(client)
            thread = threading.Thread(target=self.handle_recv, args=(client, addr))
            thread.setDaemon(True)
            thread.start()

    def handle_recv(self, client, addr):
        client.send("Connection established successfully".encode())
        while True:
            block = self.conn_pool[0].recv(4096)
            print(block)
            if len(block) != b'':
                original = pickle.loads(block, encoding="bytes")
                print(original)


if __name__ == "__main__":
    SystemServer().accept_client()

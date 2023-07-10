import pickle
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from StorageLinkedList import *

import numpy
from socket import *

import StorageLinkedList


class Server:
    def __init__(self):
        self.conn_pool = []  # connection pool
        # bind the socket to the address and port number
        self.socket_server = socket(AF_INET, SOCK_STREAM)
        host = 'localhost'  # obtain local address
        port = 2456  # set port number
        self.socket_server.bind((host, port))
        self.socket_server.listen(2)
        self.index = -1  # the index of a specific client in the connection pool

    def accept_client(self):
        while True:
            client, addr = self.socket_server.accept()  # connect establish
            print("Connection " + str(addr[0]) + " established")
            self.conn_pool.append(client)
            self.index += 1
            thread = threading.Thread(target=self.handle_recv, args=(client, addr, self.index))
            thread.setDaemon(True)
            thread.start()

    def handle_recv(self, client, addr, index):
        client.send("Connection established successfully".encode())
        data_storage = StorageLinkedList(index)
        while True:
            block = self.conn_pool[index].recv(4096)

            print(block)
            if len(block) != b'':
                original = pickle.loads(block, encoding="bytes")
                data_storage.store_data(original)  # store the data into linked list
                print(original)


if __name__ == "__main__":
    Server().accept_client()

class StorageNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def get_val(self):
        return self.val

    def get_next(self):
        return self.next

    def set_next(self, node):
        self.next = node


class StorageLinkedList:
    def __init__(self, index):
        self.dummy = StorageNode("dummy")
        self.index = index
        self.pointer = self.dummy

    def get_dummy(self):
        return self.dummy

    def store_data(self, data_block):
        self.pointer.set_next(StorageNode(data_block))
        self.pointer = self.pointer.next

    def get_pointer(self):
        return self.pointer


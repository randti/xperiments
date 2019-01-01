class Node:
    def __init__(self, value: int, next: 'Node'=None):
        self.value = value
        self.next = next
class LinkedList:
    def __init__(self):
        self.head = None

    """
    def __str__(self):
        if self.head:
            current = self.head
            p=[]
            p.append(current.value)
            while current.next != None:
                current = current.next
                p.append(current.value)
        return 'LinkedList' +' '+str(p)
    """
    def insert_before_head(self, newdata):
        new = Node(newdata)
        # Update the new nodes next val to existing node
        new.head = self.head
        self.head = new
    def add(self, x):
        if not self.head :
            self.head=Node(x, None)
        else:
            last = self.head
            while last.next != None:
                last = last.next
            last.next=Node(x,None)
    def get_simple(self):
        if self.head:
            current = self.head
            p=[]
            p.append(current.value)
            while current.next != None:
                current = current.next
                p.append(current.value)
        return p
linked = LinkedList()
n=int(input())
linked.head = Node(input())
for i in range(n-1):
    m=input()
    linked.add(m)
print(*linked.get_simple())


def reverse(cursor):
    reversed_list = None
    while cursor:
        temp = reversed_list
        next_item = cursor.next
        reversed_list = cursor
        reversed_list.next = temp
        cursor = next_item

    return reversed_list

head = Node(1, Node(2, Node(3)))



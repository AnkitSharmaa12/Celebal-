class Node:
    """Represents a node in the linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    """Manages the linked list operations."""
    def __init__(self):
        self.head = None

    def add_node(self, data):
        """Adds a node to the end of the list."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            temp = self.head
            while temp.next:
                temp = temp.next
            temp.next = new_node

    def print_list(self):
        """Prints the linked list."""
        
        if self.head is None:
            print("The list is empty.")
            return

        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("End")
        
        print()

    def delete_node(self, n):
        """Deletes the nth node (1-based index) from the list."""
        if self.head is None:
            raise Exception("Cannot delete from an empty list.")
            
        if n < 1:
            print()
            raise IndexError("Index must be 1 or higher.")

        temp = self.head

        # If head needs to be removed
        if n == 1:
            self.head = temp.next
            return

        # Find previous node of the node to be deleted
        prev = None
        count = 1
        while temp and count < n:
            prev = temp
            temp = temp.next
            count += 1

        if temp is None:
            raise IndexError("Index out of range.")

        prev.next = temp.next


# Testing the linked list

ll = LinkedList()

# Add sample data
ll.add_node(10)
ll.add_node(20)
ll.add_node(30)
ll.add_node(40)

print("Original list:")
ll.print_list()

# Deleting 1st node
try:
    ll.delete_node(2)
    print("After deleting 1st node:")
    ll.print_list()
except Exception as e:
    print(f"Error: {e}")

# Trying to delete node at index 10 (out of range)
try:
    ll.delete_node(10)
except Exception as e:
    print(f"Error: {e}")

# Trying to delete node at index 0 (invalid index)
try:
    ll.delete_node(0)
except Exception as e:
    print(f"Error: {e}")

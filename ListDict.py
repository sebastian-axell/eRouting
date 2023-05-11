from random import choice, seed

""" A Python data structure for efficient add, remove, and random.choice
 Found originally from
 https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
 """


class ListDict(object):
    def __init__(self, seed_nbr):
        self.item_to_position = {}
        self.items = []
        seed(seed_nbr)

    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove_item(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return choice(self.items)

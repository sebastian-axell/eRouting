"""
Path
========

A class to encapsulate a path together with its start and stop node, and the battery consumption to traverse the path

"""


class Path:
    def __init__(self, start, end, path, bc) -> None:
        self.start = start
        self.end = end
        self.path = path
        self.battery_consumption = bc

    def __repr__(self) -> str:
        return f"{self.start} -> {self.end}"

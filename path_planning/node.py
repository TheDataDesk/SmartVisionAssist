"""
Helper Node Object
"""

from math import sqrt
import pygame
import configuration


def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Node:
    def __init__(self, x, y, mode, start_node, end_node):
        self.x = x
        self.y = y

        self.g = int(dist(self.x, self.y, start_node.x, start_node.y))
        # print(self.g)
        self.h = int(dist(self.x, self.y, end_node.x, end_node.y))
        self.f = self.g + self.h

        # wall or node
        self.mode = mode

        self.parent = None

        self.size = 0
        self.sizeMax = configuration.Config.step_size

    def recalculate(self, a, b):
        self.g = int(dist(self.x, self.y, a.x, a.y))
        self.h = int(dist(self.x, self.y, b.x, b.y))
        self.f = self.g + self.h


class MainNode:
    def __init__(self, x, y, mode):
        self.x = x
        self.y = y

        self.pos = pygame.math.Vector2(self.x, self.y)

        self.mode = mode

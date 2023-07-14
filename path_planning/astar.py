"""
This code is an api called to find the shortest path between source and destination
by avoiding all the obstacles along the way

Standard implementation of A* path planning algorithm

Partly sourced from https://github.com/hamolicious/A-Star-Path-Finding-Algorithm
"""


import pygame
from path_planning.node import Node,MainNode
from configuration import Config as config


def find_shortest_path(start_pos, end_pos, obstacles_list):
    """
    API to find the shortest path between source & destination
    :param start_pos: position co-ordinates of source
    :param end_pos: position co-ordinates of destination
    :param obstacles_list: list of @Obstacle Obj
    :return:
    """
    print(f'Start pos - {start_pos} , end pos - {end_pos}')
    source = MainNode(start_pos[0], start_pos[1], 0)
    dest = MainNode(end_pos[0], end_pos[1], 1)

    # generate a node grid
    grid = []
    snap = []
    for i in range(0, config.floor_plan_width, config.step_size):
        for j in range(0, config.floor_plan_height, config.step_size):
            grid.append(Node(i, j, 'hidden', source, dest))
            snap.append(pygame.Rect(i, j, config.step_size, config.step_size))

    # init open and closed lists
    open_nodes = []
    closed_nodes = []
    start_node = None
    # find starting grid
    for cell in grid:
        if cell.g == 0:
            start_node = grid.index(cell)

    for node in grid:
        node.recalculate(source, dest)

        for obj in obstacles_list:
            loc = obj.rectangle
            if node.x == loc.x and node.y == loc.y:
                node.mode = 'wall'
        if node.g == 0:
            start_node = grid.index(node)

    # add starting node to open list
    open_nodes.append(grid[start_node])
    grid[start_node].mode = 'open'

    done = False
    current_node = None
    while not done:
        if len(open_nodes) == 0:
            print('No Open Nodes')

        lowestF = config.step_size ** config.step_size
        lowestH = config.step_size ** config.step_size
        for node in open_nodes:
            if node.f < lowestF:
                lowestF = node.f
                lowestH = node.h
                current_node = node

            if lowestF == node.f:
                if node.h < lowestH:
                    lowestF = node.f
                    lowestH = node.h
                    current_node = node

        # switch lists
        if current_node is not None:
            open_nodes.remove(current_node)
            closed_nodes.append(current_node)
            current_node.mode = 'closed'

        # look for neighbours
        temp = []
        for node in grid:
            if node.x == current_node.x + config.step_size and node.y == current_node.y:
                temp.append(node)

            if node.x == current_node.x - config.step_size and node.y == current_node.y:
                temp.append(node)

            if node.x == current_node.x and node.y == current_node.y + config.step_size:
                temp.append(node)

            if node.x == current_node.x and node.y == current_node.y - config.step_size:
                temp.append(node)

        # add neighbour to open list if not there and parent this node to the current node
        for node in temp:
            if closed_nodes.count(node) == 0 and node.mode != 'wall':
                if open_nodes.count(node) == 0:
                    open_nodes.append(node)
                    node.mode = 'open'

                node.parent = current_node

        # if node end is found, break
        if current_node.x == dest.x and current_node.y == dest.y:
            done = True

    path = []
    x, y = current_node.x, current_node.y
    while True:
        path.append((x + config.step_size, y + config.step_size))
        x, y = current_node.parent.x, current_node.parent.y

        current_node = current_node.parent

        if x == source.x and y == source.y:
            break
    path.append((source.x + config.step_size, source.y + config.step_size))
    path.reverse()

    return path

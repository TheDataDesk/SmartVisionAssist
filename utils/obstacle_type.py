"""
Object classes for our obstacle type
"""


class Obstacle:
    def __init__(self, rectangle, mode):
        """
        :param rectangle: Rect object of the obstacle
        :param mode: 0 - Floor plan obstacle
                     1 - Dynamic Obstacle
                     2 - Important Obstacle
        """
        self.rectangle = rectangle
        self.mode = mode

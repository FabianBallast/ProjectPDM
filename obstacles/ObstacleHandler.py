from obstacles.Obstacle import Obstacle
import numpy as np

class ObstacleHandler:
    """
    Generate an object that can deal with various obstacle manipulations.
    """

    def __init__(self, obstacleList: 'list[Obstacle]') -> None:
        """
        Initialize the handler with several obstacles.

        Args:
            objectList: list containing several objects.
        """
        self.obstacleList = obstacleList
    

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """
        Add an obstacle to the handler.

        Args:
            obstacle: Obstacle to be added to the list.
        """
        self.obstacleList.append(obstacle)

    
    def point_in_obstacle(self, point: 'tuple[float, float, float]') -> bool:
        """
        Check if a point is inside any of the obstacles.
        
        Args: 
            point: tuple of size 3 with the xyz coordinates of the point.

        Returns:
            Bool: True if in any of the obstacles.
        """
        for obstacle in self.obstacleList:
            if obstacle.point_in_obstacle([point]):
                return True

        return False

    def line_through_obstacles(self, q0, q1, n:int = 50) -> bool:
        """
        Check if the path from q0 to q1 passes through any obstacle.
        We check n points on this line to do so.

        Args: 
            - q0: initial node
            - q1: end node
            - n: number of points to evaluate on this line.

        Returns:
            Bool: True if through the obstacle.
        """
        for obstacle in self.obstacleList:
            if obstacle.line_through_obstacle(q0, q1, n):
                return True

        return False

    def plot_obstacles(self, axes) -> None:
        """
        Plot all obstacles onto the axes.

        Args:
            axes: axes to plot the obstacle onto.
        """
        for obstacle in self.obstacleList:
            if (not obstacle.dynamic):
                obstacle.plot_obstacle(axes)

    
    def get_dynamic_obstacles(self) -> np.array:
        """
        Plot all obstacles onto the axes.

        Args:
            axes: axes to plot the obstacle onto.
        """
        dynamic_list = np.array([])
        for obstacle in self.obstacleList:
            if (obstacle.dynamic):
                np.append(dynamic_list, obstacle)
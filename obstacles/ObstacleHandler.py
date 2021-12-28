from obstacles.Obstacle import Obstacle

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
            if obstacle.point_in_obstacle(point):
                return True

        return False


    def plot_obstacles(self, axes) -> None:
        """
        Plot all obstacles onto the axes.

        Args:
            axes: axes to plot the obstacle onto.
        """
        for obstacle in self.obstacleList:
            obstacle.plot_obstacle(axes)
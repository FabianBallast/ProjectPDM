import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Obstacle:
    """
    Default obstacle with generic functions for all obstacles
    """

    def __init__(self):
        """
        Basic constructor that does nothing.
        """
        self.position = None
        self.dimensions = None
        self.safety_region = None
        self.rgba = None
    
    def point_in_obstacle(self, points) -> bool:
        """
        Check if a point is inside the obstacle. Assumes a rectangular cuboid.
        Works for multiple points at the same time on the same object.

        Args: 
            - points: array of shape (N, 4) with the xyz coordinates of N points.

        Returns:
            Bool: True if in the obstacle.
        """
        return np.any(np.all(abs(self.position - points) <= self.dimensions/2 + self.safety_region, axis=1))
    
    def line_through_obstacle(self, q0, q1, n:int = 50) -> bool:
        """
        Check if the path from q0 to q1 passes through this obstacle.
        We check n points on this line to do so.

        Args: 
            - q0: initial node
            - q1: end node
            - n: number of points to evaluate on this line.

        Returns:
            Bool: True if through the obstacle.
        """
        points = np.linspace(q0, q1, n)
        return self.point_in_obstacle(points)
    
    def plot_obstacle(self, axes) -> None:
        """
        Plot this obstacle onto the axes. Assumes a rectangular cuboid.

        Args:
            axes: axes to plot the obstacle onto.
        """
        X = np.array([[[-1,  1, -1], [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1]],
                      [[-1, -1, -1], [-1, -1,  1], [ 1, -1,  1], [ 1, -1, -1]],
                      [[ 1, -1,  1], [ 1, -1, -1], [ 1,  1, -1], [ 1,  1,  1]],
                      [[-1, -1,  1], [-1, -1, -1], [-1,  1, -1], [-1,  1,  1]],
                      [[-1,  1, -1], [-1,  1,  1], [ 1,  1,  1], [ 1,  1, -1]],
                      [[-1,  1,  1], [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1]]]).astype(float)
        
        X *= self.dimensions[0:3]/2
        X += np.array(self.position[0:3])
        axes.add_collection3d(Poly3DCollection(X, facecolors=self.rgba, edgecolors = 'k'))

class Shelf(Obstacle):
    """
    Obstacle to represent (static) shelves in the warehouse.
    """

    def __init__(self, position, dimensions):
        """
        Pass the static position and dimensions to the object when initializing.

        Args:
            - position: array of size 4 in the order xyzt of the center of the obstacle.
            - dimensions: array of size 4 with the length of the obstacle in dimensions xyzt. 
        """
        self.position = position
        self.dimensions = dimensions
        self.safety_region = np.array([0.25, 0.25, 0.25, 0.1]) #Added safety for time
        self.rgba = (0, 0, 1, 0.20)



class Forklift(Obstacle):
    """
    Obstacle to represent (dynamic) forklifts in the warehouse. Not done yet.
    """
    
    def __init__(self, position, dimensions):
        """
        Pass the dynamic position over time and dimensions to the object when initializing.

        Args:
            - position: array of size 4 in the order xyzt of the center of the obstacle.
            - dimensions: array of size 4 with the length of the obstacle in dimensions xyzt. # Change
        """
        self.position = position
        self.dimensions = dimensions
        self.safety_region = np.array([0.25, 0.25, 0.25, 0.1]) #Added safety for time
        self.rgba = (0, 1, 0, 0.20)
        
        pass
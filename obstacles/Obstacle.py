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
        self.rgba = None
    
    def point_in_obstacle(self, point) -> bool:
        """
        Check if a point is inside the obstacle. Assumes a rectangular cuboid.
        
        Args: 
            - point: array of size 3 with the xyz coordinates of the point.

        Returns:
            Bool: True if in the obstacle.
        """
        return (abs(self.position - point) < self.dimensions/2).all
    
    def plot_obstacle(self, axes) -> None:
        """
        Plot this obstacle onto the axes. Assumes a rectangular cuboid.

        Args:
            axes: axes to plot the obstacle onto.
        """
        # x_max, y_max, z_max = np.round(self.position + self.dimensions/2).astype(int)
        # x_min, y_min, z_min = np.round(self.position - self.dimensions/2).astype(int)

        # x, y, z = np.indices((x_max, y_max, z_max))
        # obstacle = (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max) & (z_min <= z) & (z <= z_max)
        # axes.voxels(obstacle, facecolor=self.rgba)

        X = np.array([[[-1,  1, -1], [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1]],
                      [[-1, -1, -1], [-1, -1,  1], [ 1, -1,  1], [ 1, -1, -1]],
                      [[ 1, -1,  1], [ 1, -1, -1], [ 1,  1, -1], [ 1,  1,  1]],
                      [[-1, -1,  1], [-1, -1, -1], [-1,  1, -1], [-1,  1,  1]],
                      [[-1,  1, -1], [-1,  1,  1], [ 1,  1,  1], [ 1,  1, -1]],
                      [[-1,  1,  1], [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1]]]).astype(float)
        
        X *= self.dimensions/2
        X += np.array(self.position)
        print(self.rgba)
        axes.add_collection3d(Poly3DCollection(X, facecolors=self.rgba, edgecolors = 'k'))

class Shelf(Obstacle):
    """
    Obstacle to represent (static) shelves in the warehouse.
    """

    def __init__(self, position, dimensions):
        """
        Pass the static position and dimensions to the object when initializing.

        Args:
            - position: array of size 3 in the order xyz of the center of the obstacle.
            - dimensions: array of size 3 with the length of the obstacle in dimensions xyz. 
        """
        self.position = position
        self.dimensions = dimensions
        self.rgba = (1, 0, 0, 0.5)



class Forklift(Obstacle):
    """
    Obstacle to represent (dynamic) forklifts in the warehouse. Not done yet.
    """
    
    def __init__(self, position, dimensions):
        """
        Pass the dynamic position over time and dimensions to the object when initializing.

        Args:
            - position: array of size 3 in the order xyz of the center of the obstacle.
            - dimensions: array of size 3 with the length of the obstacle in dimensions xyz. # Change
        """
        pass
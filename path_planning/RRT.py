from obstacles.ObstacleHandler import ObstacleHandler
from path_planning.Tree import Tree
import numpy.random as rand

class RRT:
    """
    Class to plan a path using basic RRT.
    """
    def __init__(self, max_configuration_space, obsHand: ObstacleHandler, seed: int = 4715527):
        """
        Initialize the RRT with the maximum size of the configuration space.
        This is in R^3 for now. Furthermore, also add the obstacles.

        Args:
            - max_configuration_space: Numpy array with maximum xyz dimensions.
            - obsHand: Obstacle handler with obstacles.
        """
        self.tree = None
        self.max_conf_space= max_configuration_space
        self.obstacleHandler = obsHand
        rand.seed(seed=seed)

    def find_path(self, q_0, q_goal, n: int) -> Tree:
        """
        Find the path from q_0 to q_goal in a maximum of n steps.
        For now, all configurations are in R^3 (no yaw, pitch and roll).

        Args:
            - q_0: Initial configuration.
            - q_goal: Target configuration.
            - n: Maximum number of iterations.

        Returns:
            - Edges of the tree.
            - Vertices of the tree.
        """
        self.tree = Tree(q_0)

        for i in range(n):
            q_random = rand.uniform(high=self.max_conf_space)

            while self.obstacleHandler.point_in_obstacle(q_random):
                q_random = rand.uniform(high=self.max_conf_space)

            q_closest = self.tree.find_closest_neighbour(q_random)

            if not self.obstacleHandler.line_through_obstacles(q_random, q_closest):
                self.tree.add_vertex(q_random, q_closest)

                if not self.obstacleHandler.line_through_obstacles(q_random, q_goal):
                    self.tree.add_final_vertex(q_goal, q_random)

                    return self.tree
        
        return self.tree
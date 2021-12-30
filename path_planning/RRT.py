from obstacles.ObstacleHandler import ObstacleHandler
from path_planning.Tree import Tree
import numpy.random as rand
from path_planning.Vertex import Vertex
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
        self.max_conf_space = max_configuration_space
        self.obstacleHandler = obsHand
        rand.seed(seed=seed)

    def find_path(self, x_0, x_goal, n: int) -> Tree:
        """
        Find the unoptimal path from q_0 to q_goal in a maximum of n steps.
        For now, all configurations are in R^3 (no yaw, pitch and roll).

        Args:
            - x_0: Initial state.
            - x_goal: Target state.
            - n: Maximum number of iterations.

        Returns:
            - Tree
        """
        self.tree = Tree(x_0)
        q_goal = Vertex(x_goal, 0)
        goal_added_to_tree = False

        for i in range(n):
            q_random = Vertex(rand.uniform(high=self.max_conf_space), 0)

            while self.obstacleHandler.point_in_obstacle(q_random.state):
                q_random = Vertex(rand.uniform(high=self.max_conf_space), 0)

            collision_free_neighbours = self.tree.find_collision_free_neighbours(q_random, self.tree.vertices, self.obstacleHandler)

            if len(collision_free_neighbours) > 0:
                q_closest = self.tree.find_closest_neighbour(q_random, collision_free_neighbours)
                self.tree.add_vertex(q_random, q_closest)

                if not self.obstacleHandler.line_through_obstacles(q_random.state, q_goal.state) and not goal_added_to_tree:
                    print("Goal found!")
                    self.tree.add_vertex(q_goal, q_random)
                    goal_added_to_tree = True
        
        if goal_added_to_tree:
            self.tree.sort(q_goal)
        else:
            self.tree.sort()
            
        return self.tree
    
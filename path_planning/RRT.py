from numpy.lib.shape_base import expand_dims
from obstacles.ObstacleHandler import ObstacleHandler
from path_planning.Tree import Tree
import numpy.random as rand
from path_planning.Vertex import Vertex
from path_planning.TrajectoryOptimization import min_time
class RRT:
    """
    Class to plan a path using basic RRT.
    """
    def __init__(self, max_configuration_space, obsHand: ObstacleHandler, seed: int = 4715527):
        """
        Initialize the RRT with the maximum size of the configuration space.
        This is in R^4 for now. Furthermore, also add the obstacles.

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
        For now, all configurations are in R^4 (no yaw, pitch and roll).

        Args:
            - x_0: Initial state.
            - x_goal: Target state.
            - n: Maximum number of iterations.

        Returns:
            - Tree
        """
        self.tree = Tree(x_0)
        q_goal = Vertex(x_goal)
        goal_added_to_tree = False

        for i in range(n):

            # Find random point not in obstacle
            q_random = Vertex(rand.uniform(high=self.max_conf_space))

            while self.obstacleHandler.point_in_obstacle(q_random.state):
                q_random = Vertex(rand.uniform(high=self.max_conf_space))

            # Find its neighbours that it can reach
            # Take care that you can only move in positive time...
            _, collision_free_neighbours, _ = self.tree.find_collision_free_neighbours(q_random, self.tree.vertices, self.obstacleHandler)
            # print(len(collision_free_neighbours))
            # If any neighbour is reachable, find the closest, add it to the tree and check if the goal can be reached from there.
            if len(collision_free_neighbours) > 0:
                q_closest = self.tree.find_closest_neighbour(q_random, collision_free_neighbours)
                self.tree.add_vertex(q_random, q_closest)

                min_t_to_goal = min_time(q_random.state[0:3], q_goal.state[0:3])
                goal_time_pass = q_goal.state[3] - q_random.state[3] > min_t_to_goal
                if not self.obstacleHandler.line_through_obstacles(q_random.state, q_goal.state) and not goal_added_to_tree and goal_time_pass:
                    print(f"Goal found in {i} iterations!")
                    self.tree.add_vertex(q_goal, q_random)
                    q_goal.state[3] = q_random.state[3] + 1.1 * min_t_to_goal
                    goal_added_to_tree = True
                    break
        
        # Sort the tree such that we know the path from start to end.
        if goal_added_to_tree:
            self.tree.sort(q_goal)
        else:
            raise Exception("Goal not found...")

        return self.tree
    
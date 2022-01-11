import numpy as np
from path_planning.Vertex import Vertex

class Tree:
    """
    Create a tree to perform operations on.
    """
    def __init__(self, x_0):
        """
        Initialize the tree with an initial state.

        Args:
            - x_0: State of first vertex.
        """
        self.vertices = [Vertex(x_0, True)]
        self.sorted_vertices = []

    def find_closest_neighbour(self, q: Vertex, neighbours=None):
        """
        Find the closest neighbour of q in the tree.

        Args:
            - q: Vertex for which we find the closest neighbour.
            - neighbours: Neighbours to check for. If None, take all.

        Returns:
            - q_closest: Closest neighbour.
        """
        if neighbours is None:
            neighbours=self.vertices

        distances = [q.distance_to(vertex) for vertex in neighbours]

        return neighbours[np.argmin(distances)]
    
    def find_lowest_cost_neighbour(self, q, neighbours=None):
        """
        Find the neighbour of q with the lowest cost (e.g., closest to root) in the tree.

        Args:
            - q: Vertex for which we find the closest neighbour.
            - neighbours: Neighbours to check for. If None, take all.

        Returns:
            - q_lowest_cost: Lowest cost neighbour.
            - lowest_cost: Lowest cost
        """
        if neighbours is None:
            neighbours=self.vertices

        costs = [q.find_cost_from(vertex) for vertex in neighbours]

        return neighbours[np.argmin(costs)], np.min(costs)
    
    def find_collision_free_neighbours(self, q_center, q_neighbours, obs_hand):
        """
        Find the neighbours of q_center from which the connections do not pass through obstacles.

        Args:
            q_center: Vertex that is the center of all edges.
            q_neighbours: List of vertices with all neighbours.
            obs_hand: Obstacle handler with all obstacles.
        
        Returns:
            List with all collision-free neighbours.
        """
        q_neighbours = [q for q in q_neighbours if not obs_hand.line_through_obstacles(q.state, q_center.state)]
        q_neighbours_before_current = [q for q in q_neighbours if q.state[3] - q_center.state[3] < 0]
        q_neighbours_after_current = [q for q in q_neighbours if q.state[3] - q_center.state[3] > 0]

        return q_neighbours, q_neighbours_before_current, q_neighbours_after_current

    def add_vertex(self, q_add, q_connection_point=None) -> None:
        """
        Add a vertex q_add to the tree. 
        If no connecting vertex is given, we connect it to the closest vertex.

        Args:
            - q_add: Vertex which is added to the tree.
            - q_connection_point: Vertex to which q_add is connected.
        """
        self.vertices.append(q_add)

        if q_connection_point is None:
            q_connection_point = self.find_closest_neighbour(q_add)

        q_add.make_edge_with_parent(q_connection_point)

    def sort(self, q_goal=None) -> None:
        """
        Sort the tree. Go from q_goal to start through the tree, then reverse to get path from start to goal.

        Args:
            - q_goal: Goal vertex. If not given, assumed to be last value in list.
        """
        if q_goal is None:
            self.sorted_vertices.append(self.vertices[-1])
        else:
            self.sorted_vertices.append(q_goal)

        while self.sorted_vertices[-1].parent_vertex is not None:
            self.sorted_vertices.append(self.sorted_vertices[-1].parent_vertex)
        
        self.sorted_vertices.reverse()

    def plot_tree(self, axes) -> None:
        """
        Plot the tree onto the axes.

        Args: 
            - axes: Axes to plot the tree onto.
        """
        # for vertex in self.vertices:
        #     vertex.draw_edge_to_parent(axes)
        
        path_taken = np.asarray([vertex.state for vertex in self.sorted_vertices])
        axes.plot(path_taken[:, 0], path_taken[:, 1], path_taken[:, 2], 'go-', zorder=2, label="Path")
        axes.plot([path_taken[ 0][0]], [path_taken[ 0][1]], [path_taken[ 0][2]], 'bo', zorder=3, label="Start")
        axes.plot([path_taken[-1][0]], [path_taken[-1][1]], [path_taken[-1][2]], 'yo', zorder=3, label="End")

    def reroute(self, vertex_add, neighbours, obs_hand, gamma=10) -> None:
        """
        Reroute the tree after adding a vertex. 

        Args:
            - vertex_add: vertex that is added.
            - neighbours: Collision-free neighbours of vertex_add.
            - obs_hand: Obstacle handler with obstacles.
            - gamma: gamma from equation in the slides.
        """
        # Calculate radius from slides (note: there is a smaller one in one of the slides.)
        n = len(self.vertices)
        d = len(self.vertices[0].state)
        radius = gamma * (np.log(n) / n)**(1/d)

        # Check the vertices that are within this radius
        vertices_to_check = [vertex for vertex in neighbours if vertex_add.distance_to(vertex) <= radius]
        vertices_to_check_before_current = [vertex for vertex in vertices_to_check if vertex_add.state[3] - vertex.state[3] > 0]

        if len(vertices_to_check_before_current) > 0:
            
            closest_neighbour = self.find_closest_neighbour(vertex_add, vertices_to_check_before_current)
            lowest_cost_neighbour, _ = self.find_lowest_cost_neighbour(vertex_add, vertices_to_check_before_current)

            # If the lowest cost neighbour is different from the closest, rerouting might be needed. Otherwise, it is a general addition.
            if not lowest_cost_neighbour == closest_neighbour:
                
                # Check all vertices for rerouting, and add default vertex to end to recheck that one at the end.
                vertices_to_check.append(vertex_add)
            for vertex in vertices_to_check:
                # For each vertex only check the ones which are earlier in time.
                q_neighbours, collision_free_neighbours, q_neighbours_after_current = self.find_collision_free_neighbours(vertex, vertices_to_check, obs_hand)
                if len(collision_free_neighbours) > 0:
                    lowest_cost_neighbour, lowest_cost = self.find_lowest_cost_neighbour(vertex, collision_free_neighbours)

                    if lowest_cost < vertex.get_cost() and not lowest_cost_neighbour == vertex.parent_vertex: 
                        vertex.make_edge_with_parent(lowest_cost_neighbour)

    




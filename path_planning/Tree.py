import numpy as np
import scipy.spatial.distance as distance
import numpy as np

class Tree:
    """
    Create a tree to perform operations on.
    """
    def __init__(self, q0):
        """
        Initialize the tree with an initial vertex.

        Args:
            - q0: Initial vertex.
        """
        self.vertices = [q0]
        self.edges  = []
        self.sorted_vertices = []

    def find_closest_neighbour(self, q):
        """
        Find the closest neighbour of q in the tree.

        Args:
            - q: Vertex for which we find the closest neighbour.

        Returns:
            - q_closest: Closest neighbour.
        """
        distances = distance.cdist([q], self.vertices)

        return self.vertices[np.argmin(distances)]

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

        self.edges.append((q_connection_point, q_add))

    def add_final_vertex(self, q_goal, q_connection_point=None) -> None:
        """
        Add the final vertex q_add to the tree. 
        If no connecting vertex is given, we connect it to the closest vertex.
        This also finds the path from start to goal.

        Args:
            - q_goal: Vertex which is added to the tree.
            - q_connection_point: Vertex to which q_add is connected.
        """
        self.add_vertex(q_goal, q_connection_point)

        q_to_add = q_goal
        self.sorted_vertices.append(q_to_add)

        while np.all(q_to_add != self.vertices[0]):
            for edge in self.edges:
                if np.all(edge[1] == q_to_add):
                    q_to_add = edge[0]
                    self.sorted_vertices.append(q_to_add)
        
        self.sorted_vertices.reverse()

    def plot_tree(self, axes) -> None:
        """
        Plot the tree onto the axes.

        Args: 
            - axes: Axes to plot the tree onto.
        """

        for edge in self.edges:
            start, end = edge
            axes.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'ro-', zorder=1)
        
        sorted_vertices = np.asarray(self.sorted_vertices)
        axes.plot(sorted_vertices[:, 0], sorted_vertices[:, 1], sorted_vertices[:, 2], 'go-', zorder=2)
        axes.plot([self.vertices[ 0][0]], [self.vertices[ 0][1]], [self.vertices[ 0][2]], 'bo', zorder=3)
        axes.plot([self.vertices[-1][0]], [self.vertices[-1][1]], [self.vertices[-1][2]], 'yo', zorder=3)
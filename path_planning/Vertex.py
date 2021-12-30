import numpy as np

class Vertex:
    """
    Create a vertex object to perform operations on.
    """
    def __init__(self, state, cost: float) -> None:
        """
        Initialize the vertex with a state and a cost to reach that vertex.

        Args:
            - state: Numpy array with 3 elements containing the state.
            - cost: Cost to reach the state.
        """
        self.state = state
        self.cost = cost
        self.parent_vertex = None
    
    def change_cost(self, cost: float) -> None:
        """
        Update the cost of this vertex.

        Args:
            - cost: New cost of this vertex.
        """
        self.cost = cost
    
    def find_cost_from(self, other) -> float:
        """
        Find the cost of this vertex if the other was the parent.

        Args:
            - other: Possible parent vertex.
        """
        return self.distance_to(other) + other.cost
    
    def set_parent_vertex(self, parent) -> None:
        """
        Set the cost of its current parent vertex.
        """
        self.parent_vertex = parent

    def distance_to(self, other) -> float:
        """
        Return the distance of this vertex to another vertex.

        Args:
            - other: New vertex.
        
        Returns:
            Euclidean distance to vertex.
        """
        return np.linalg.norm(self.state - other.state)
    
    def draw_edge_to_parent(self, axes, draw_type='ro-', zorder=1) -> None:
        """
        Draw the edge from this vertex to parent vertex.

        Args:
            - axes: Axes to plot the axes onto.
            - draw_type: How to draw the edge. Color, line style etc.
            - zorder: At which level to draw the edge. 
        """
        if self.parent_vertex is not None:
            axes.plot([self.state[0], self.parent_vertex.state[0]], 
                      [self.state[1], self.parent_vertex.state[1]], 
                      [self.state[2], self.parent_vertex.state[2]], draw_type, zorder=zorder)

    def __eq__(self, other) -> bool:
        """
        Comparison with ==
        """
        return np.all(self.state == other.state)

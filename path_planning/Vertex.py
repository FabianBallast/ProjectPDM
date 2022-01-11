import numpy as np

class Vertex:
    """
    Create a vertex object to perform operations on.
    """
    def __init__(self, state, root: bool=False) -> None:
        """
        Initialize the vertex with a state and a cost to reach that vertex.

        Args:
            - state: Numpy array with 4 elements containing the state.
            - root: If this vertex is the root of the tree.
        """
        self.state = state
        self.root = root
        self.parent_vertex = None
        self.child_vertices = []
    
    def get_cost(self) -> float:
        """
        Return the cost to reach this vertex.
        """
        if self.root:
            return 0
        else:
            return self.find_cost_from(self.parent_vertex)
    
    def find_cost_from(self, other) -> float:
        """
        Find the cost of this vertex if the other was the parent.

        Args:
            - other: Possible parent vertex.
        """
        return self.distance_to(other) + other.get_cost()
    
    def make_edge_with_parent(self, parent) -> None:
        """
        Make edge with parent. E.g., update child and parent vertices of both vertices.

        Args:
            - parent: 
        """

        if self.parent_vertex is None:
            self.parent_vertex = parent
        else:
            self.parent_vertex.remove_child(self)
            self.parent_vertex = parent

        parent.child_vertices.append(self)

    def remove_child(self, child) -> None:
        """
        Remove a child vertex from the list of children.

        Args:
            - child: Child vertex to be removed.
        """ 
        self.child_vertices.remove(child)

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

    def distance_to_root(self) -> float:
        """
        Draw the edge from this vertex to parent vertex.

        Args:
            - axes: Axes to plot the axes onto.
            - draw_type: How to draw the edge. Color, line style etc.
            - zorder: At which level to draw the edge. 
        """
        if self.parent_vertex is not None:
            cost = np.linalg.norm(self.state[0:3] - self.parent_vertex.state[0:3])
            return self.get_cost() + self.parent_vertex.distance_to_root()
        return 0

    def __eq__(self, other) -> bool:
        """
        Comparison with ==
        """
        return np.all(self.state == other.state)
    
    def __str__(self) -> str:
        """
        Print info of Vertex.
        """
        if self.parent_vertex is None:
            return f"Vertex with state: {self.state}, \t cost: {self.get_cost()}, and no parent." 
        else:
            return f"Vertex with state: {self.state}, \t cost: {self.get_cost()}, and a parent at {self.parent_vertex.state} with cost {self.parent_vertex.get_cost()}." 

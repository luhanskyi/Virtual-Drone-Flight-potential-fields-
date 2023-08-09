import sys
import numpy as np
from matplotlib import pyplot as plt
from pathfinding import *
from enum import Enum
import math


def squareFunction(x, y, obstacles, potential):
    """
    Calculates the potential field for obstacle avoidance.

    Args:
    x (float): x-coordinate of the current position.
    y (float): y-coordinate of the current position.
    obstacles (list): List of obstacles.

    Returns:
        float: Total potential of the potential field.
    """

    totalPotential = 0.0

    for obstacle in obstacles:
        r1 = obstacle.getRadius()
        r2 = obstacle.getRadius() + 4
        distance = math.sqrt((x - obstacle.getX()) ** 2 + (y - obstacle.getY()) ** 2)

        if distance <= r1:
            return 0
        elif r1 < distance < r2:
            totalPotential += (potential - 1) / (distance - r1 + 1)
        elif distance >= r2:
            totalPotential += 0

    if totalPotential > potential:
        totalPotential = potential

    return totalPotential


def squareFunction3D(x, y, z, obstacles, potential):
    """
    Calculates the potential field for obstacle avoidance in 3D.

    Args:
        x (float): x-coordinate of the current position.
        y (float): y-coordinate of the current position.
        z (float): z-coordinate of the current position.
        obstacles (list): List of obstacles.
        potential (float): Maximum potential value.

    Returns:
        float: Total potential of the potential field.
    """

    totalPotential = 0.0

    for obstacle in obstacles:
        r1 = obstacle.getRadius()
        r2 = obstacle.getRadius() + 4
        distance = math.sqrt((x - obstacle.getX()) ** 2 + (y - obstacle.getY()) ** 2 + (z - obstacle.getZ()) ** 2)

        if distance <= r1:
            return 0
        elif r1 < distance < r2:
            totalPotential += (potential - 1) / (distance - r1 + 1)
        elif distance >= r2:
            totalPotential += 0

    if totalPotential > potential:
        totalPotential = potential

    return totalPotential


def gaussFunction(x, y, obstacles, potential):
    """
    Calculates the potential field using the Gaussian function for obstacle avoidance.

    Args:
        x (float): x-coordinate of the current position.
        y (float): y-coordinate of the current position.
        obstacles (list): List of obstacles.
        potential (float): Maximum potential value.

    Returns:
        float: Total potential of the potential field.
    """
    totalPotential = 0.0

    for obstacle in obstacles:
        radius2 = obstacle.getRadius() + 4
        distance = math.sqrt((x - obstacle.getX()) ** 2 +
                             (y - obstacle.getY()) ** 2)

        if distance <= obstacle.getRadius():
            return 0
        elif obstacle.getRadius() < distance < radius2:
            totalPotential += (potential - 1) / (distance - obstacle.getRadius() + 1)

    if totalPotential > potential:
        totalPotential = potential

    return totalPotential



def gaussFunctionUpdated(x, y, obstacles, potential):
    """
    Calculates the updated Gaussian potential field for obstacle avoidance.

    Args:
        x (float): x-coordinate of the current position.
        y (float): y-coordinate of the current position.
        obstacles (list): List of obstacles.
        potential (float): Maximum potential value.

    Returns:
        float: Total potential of the updated Gaussian potential field.
    """

    totalPotential = 0.0

    for obstacle in obstacles:
        radius2 = obstacle.getRadius() + 3
        distance = math.sqrt((x - obstacle.getX()) ** 2 + (y - obstacle.getY()) ** 2)

        if distance <= obstacle.getRadius():
            return potential
        elif obstacle.getRadius() < distance <= radius2:
            sigma = (radius2 - obstacle.getRadius()) / 3  # Adjust the standard deviation according to the range of the Gaussian function.
            exponent = -((distance - obstacle.getRadius()) ** 2) / ((2 * sigma )** 2)
            gaussian = math.exp(exponent)
            totalPotential += gaussian * potential

    if totalPotential > potential:
        totalPotential = potential

    return totalPotential

def gaussFunctionUpdated3D(x, y, z, obstacles, potential):
    """
    Calculates the updated Gaussian potential field for obstacle avoidance.
    Args:
        x (float): x-coordinate of the current position.
        y (float): y-coordinate of the current position.
        z (float): z-coordinate of the current position.
        obstacles (list): List of obstacles.
        potential (float): Maximum potential value.

    Returns:
        float: Total potential of the updated Gaussian potential field.

    """

    totalPotential = 0.0

    for obstacle in obstacles:
        radius2 = obstacle.getRadius() + 3
        distance = math.sqrt((x - obstacle.getX()) ** 2 + (y - obstacle.getY()) ** 2 + (z - obstacle.getZ()) ** 2)

        if distance <= obstacle.getRadius():
            return potential
        elif obstacle.getRadius() < distance <= radius2:
            sigma = (radius2 - obstacle.getRadius()) / 3
            exponent = -((distance - obstacle.getRadius()) ** 2) / ((2 * sigma) ** 2)
            gaussian = math.exp(exponent)
            totalPotential += gaussian * potential

    if totalPotential > potential:
        totalPotential = potential

    return totalPotential

def exponentialFunction(x, y, obstacles, potential):
    """
    Calculates the potential field for repelling obstacles with exponential growth.

    Args:
        x (float): x-coordinate of the current position.
        y (float): y-coordinate of the current position.
        obstacles (list): List of obstacles.
        potential (int): maximum value of force

    Returns:
        float: total potential of the potential field.

    """

    totalPotential = 0.0

    for obstacle in obstacles:
        r1 = obstacle.getRadius()
        r2 = obstacle.getRadius() + 3
        distance = math.sqrt((x - obstacle.getX()) ** 2 + (y - obstacle.getY()) ** 2)

        if distance <= r1:
            totalPotential += 1.0  #  Using 1 as the base value for exponential growth
        elif r1 < distance < r2:
            exponential_factor = math.exp((r1 - distance) / 2)  # Exponential growth factor
            totalPotential += exponential_factor

    if totalPotential > potential:
        totalPotential = potential

    return totalPotential


class ForceCalculation(Enum):
    """
    This class represents the force calculation type for the pathfinding algorithm.
    """
    SQUARE = 1
    GAUSSIAN = 2


class PfmPathfinding(Pathfinding):
    """
    This class represents the pathfinding algorithm for finding a path from a start cell to an end cell in a grid
    using pfm Algorithm.

    Attributes:
        startCell: The start cell of the pathfinding algorithm.
        endCell: The end cell of the pathfinding algorithm.
        maxHeight: The maximum allowed height for the pathfinding algorithm.
        queue: A list representing the queue of cells to be visited.
        visited: A list representing the cells that have already been visited.
        parents: A dictionary representing the parents of each cell in the path.
        path: A list representing the final path found by the algorithm.
        obstacles: A list with all obstacles.
    """

    def __init__(self, startCell: GridCell, endCell: GridCell, maxHeight: int, grid: Grid, obstacles: [Obstacle]):
        """
        Initialize a new instance of the Pfm_Pathfinding class.

        Args:
            startCell: The start cell of the pathfinding algorithm.
            endCell: The end cell of the pathfinding algorithm.
            maxHeight: The maximum allowed height for the pathfinding algorithm.
            obstacles: A list with all obstacles.
        """
        super().__init__(startCell, endCell, maxHeight, grid, obstacles)
        self.addForce = False
        self.potential = 255000000


    def run(self) -> [GridCell]:
        """
        Run the pfm pathfinding algorithm.

        Returns:
            A list representing the final path found by the algorithm.
        """
        #self.visualizePotentialFields(self.obstacles, resolution=0.1)
        self.solve()
        for cell in self.visited:
            cell.open = False
            cell.closed = False
        self.queue.clear()
        self.visited.clear()
        self.addForce = True
        self.solve()
        if self.endCell.isClosed():
            self.reconstruct()
        # dont use if you enable diagonal function
        # self.path = super().optimizePath(self.path)
        self.creatJSON()

        return self.path

    def visualizePotentialFields(self, obstacles, resolution=0.1):
        """
        Visualizes the potential fields around obstacles in 2D and 3D.

        Args:
            obstacles (list): List of obstacles.
            resolution (float): Resolution of the grid.

        Returns:
            None
        """

        # Add a buffer to the grid range
        buffer = 20.0

        # Define the range of the grid with a buffer
        min_x = min([obstacle.getX() - obstacle.getRadius() for obstacle in obstacles]) - buffer
        max_x = max([obstacle.getX() + obstacle.getRadius() for obstacle in obstacles]) + buffer
        min_y = min([obstacle.getY() - obstacle.getRadius() for obstacle in obstacles]) - buffer
        max_y = max([obstacle.getY() + obstacle.getRadius() for obstacle in obstacles]) + buffer

        # Create a grid with given resolution
        x_range = np.arange(min_x, max_x + resolution, resolution)
        y_range = np.arange(min_y, max_y + resolution, resolution)
        grid_x, grid_y = np.meshgrid(x_range, y_range)

        # Calculate the potential field for each point on the grid
        potential_field = np.zeros_like(grid_x)
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                potential_field[j, i] = gaussFunction(x, y, obstacles,
                                                      self.potential)  # Use your potential function here

        # Plot the potential field in 2D
        plt.figure(figsize=(10, 8))
        plt.contourf(grid_x, grid_y, potential_field, cmap='hot')
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Potential Field')
        plt.scatter([obstacle.getX() for obstacle in obstacles], [obstacle.getY() for obstacle in obstacles],
                    color='red', marker='o')
        plt.gca().invert_yaxis()  # Invert the y-axis
        plt.show()

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the potential field as a surface
        ax.plot_surface(grid_x, grid_y, potential_field, cmap='hot')

        # Add scatter plot for obstacles
        ax.scatter([obstacle.getX() for obstacle in obstacles], [obstacle.getY() for obstacle in obstacles],
                   [gaussFunction(obstacle.getX(), obstacle.getY(), obstacles, self.potential) for obstacle in
                    obstacles],
                   color='red', marker='o')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Potential')
        ax.set_title('Potential Field')
        plt.gca().invert_yaxis()  # Invert the y-axis
        plt.show()

    def solve(self):
        """
        Solve the pathfinding algorithm.
        """
        self.endCell.setForce(0)
        self.queue.append(self.endCell)
        self.endCell.setOpen(True)
        # iterates above all possibilities and add new possibilities
        while len(self.queue) > 0:
            cell: GridCell = self.queue.pop(0)
            cell.setOpen(False)
            nextForce = (cell.getForce() + 1)
            if not self.addForce:
                self.visited.append(cell)
            cell.setClosed(True)
            neighbours: [GridCell] = cell.getNeighborCells()
            self.setForceFromNeighbour(cell, neighbours, nextForce, ForceCalculation.SQUARE)

    def setForceFromNeighbour(self, cell: GridCell, neighbours: [GridCell], nextForce: int, funct: ForceCalculation):
        """
        The method sets the height from the next neighbours and add the actual cell
        to the parents from the neighbours
        Args:
            cell: the actual cell of the iteration
            neighbours: the next cell from the actual cell
            nextForce: the force which should get the neighbours
            funct: the used function to calculate the height
        """
        for neighbour in neighbours:
            if neighbour.getForce() < self.potential and (not neighbour.isClosed()):
                if not neighbour.isOpen():
                    # set height of the next higher value of the actual cell
                    if self.addForce:
                        if funct == ForceCalculation.GAUSSIAN:
                            force = gaussFunctionUpdated(
                                neighbour.getPositionX(), neighbour.getPositionY(), self.obstacles, self.potential)
                        elif funct == ForceCalculation.SQUARE:
                            force = squareFunction(
                                neighbour.getPositionX(), neighbour.getPositionY(), self.obstacles, self.potential)
                        neighbour.setForce(neighbour.getForce() + force)
                    else:
                        neighbour.setForce(nextForce)
                    self.parents[neighbour] = cell
                if (not neighbour.isOpen()) and (not neighbour.isClosed()):
                    self.queue.append(neighbour)
                    neighbour.setOpen(True)

    def reconstruct(self):
        """
        Construct the path and allways pick the cell with the lowest height.
        """
        current_cell: GridCell = self.startCell

        if current_cell not in self.parents:
            return
        while current_cell != self.endCell:
            self.path.append(current_cell)
            next_cell = self.getLowestForce(current_cell)
            if current_cell != next_cell:
                current_cell = next_cell
            else:
                self.path.clear()
                return
            # comment this out if you will the cells with the set heights
            print("Cell ", current_cell.getForce(), " x ", current_cell.getPositionX(), " y ",
            current_cell.getPositionY())

        self.path.append(self.endCell)

    def getLowestForce(self, cell: GridCell) -> [GridCell]:
        neighbours: [GridCell] = cell.getNeighborCells()
        change = False
        for neighbour in neighbours:
            if (neighbour.getForce() >= self.potential) or (neighbour in self.path):
                continue
            if cell.getForce() > neighbour.getForce():
                cell = neighbour
                change = True
        if not change:
            for neighbour in neighbours:
                if (neighbour.getForce() >= self.potential) or (neighbour in self.path):
                    continue
                if (cell.getForce() >= neighbour.getForce()) or cell in self.path:
                    cell = neighbour
        return cell

def test_squareFunction():
    """
    r2 = obstacle.getRadius() + 4
    """

    # Test Case 1: No obstacles, expected total potential is 0.0
    obstacles = []
    x = 0
    y = 0
    expected_result = 0.0
    actual_result = squareFunction(x, y, obstacles, 255000000)
    assert math.isclose(actual_result, expected_result, abs_tol=1e-6), f"Test Case 1 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 1 passed successfully.")

    # Test Case 2: Obstacle outside the range, expected total potential is 0.0
    obstacles = [Obstacle(5, 5, 2)]
    x = 0
    y = 0
    expected_result = 0.0
    actual_result = squareFunction(x, y, obstacles, 255000000)
    assert math.isclose(actual_result, expected_result, abs_tol=1e-6), f"Test Case 2 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 2 passed successfully.")

    # Test Case 3: The drone is extremely close to the obstacle, and the potential field reaches nearly the maximum value.
    obstacles = [Obstacle(3, 2, 1)]
    x = 4.0000000000001
    y = 2
    expected_result = 255000000
    actual_result = squareFunction(x, y, obstacles, 255000000)
    assert math.isclose(actual_result, expected_result, rel_tol=0.1), f"Test Case 3 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 3 passed successfully.")

    # Test Case 4: The drone just enters the range of the potential field.
    obstacles = [Obstacle(3, 2, 1)]
    x = 7.99999999
    y = 2
    expected_result = 60000000
    actual_result = squareFunction(x, y, obstacles, 255000000)
    assert math.isclose(actual_result, expected_result, rel_tol=0.2), f"Test Case 4 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 4 passed successfully.")


def test_squareFunction():
    # Test Case 1: No obstacles
    obstacles = []
    x = 0
    y = 0
    expected_result = 0.0
    actual_result = squareFunction(x, y, obstacles)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 1 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 1 passed successfully.")

    # Test Case 2: One obstacle, drone far away
    obstacles = [Obstacle(5, 5, 2)]
    x = 0
    y = 0
    expected_result = 0.0
    actual_result = squareFunction(x, y, obstacles)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 2 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 2 passed successfully.")

    # Test Case 3: Obstacle at coordinates (5, 5) with radius 2, drone inside the obstacle
    obstacles = [Obstacle(5, 5, 2)]
    x = 5
    y = 5
    expected_result = 1.0
    actual_result = squareFunction(x, y, obstacles)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 3 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 3 passed successfully.")

    # Test Case 4: Drone directly in front of the obstacle
    obstacles = [Obstacle(3, 2, 1)]
    x = 5
    y = 2
    expected_result = 0.60653
    actual_result = squareFunction(x, y, obstacles)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 4 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 4 passed successfully.")

    # Test Case 5: List with multiple obstacles, each influencing the drone
    obstacles = [Obstacle(3, 2, 1), Obstacle(6, 6, 1), Obstacle(10, 2, 1)]
    x = 3
    y = 2
    expected_result = 1.0
    actual_result = squareFunction(x, y, obstacles)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 5 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 5 passed successfully.")


def test_gaussFunction():
    # Test Case 1: No obstacles
    obstacles = []
    x = 0
    y = 0
    expected_result = 0.0
    actual_result = gaussFunction(x, y, obstacles, 255000000)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 1 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 1 passed successfully.")

    # Test Case 2: One obstacle, drone far away
    obstacles = [Obstacle(5, 5, 2)]
    x = 0
    y = 0
    expected_result = 0.0
    actual_result = gaussFunction(x, y, obstacles, 255000000)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 2 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 2 passed successfully.")

    # Test Case 3: Drone right in front of the obstacle
    obstacles = [Obstacle(3, 2, 2)]
    x = 5
    y = 2
    expected_result = 255000000  # Potential
    actual_result = gaussFunction(x, y, obstacles, 255000000)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 3 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 3 passed successfully.")

    # Test Case 4: Drone far away from the obstacle
    obstacles = [Obstacle(6, 2, 2)]
    x = 3
    y = 2
    expected_result = 198594199.7  # Potential
    actual_result = gaussFunction(x, y, obstacles, 255000000)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 4 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 4 passed successfully.")

    # Test Case 5: List of multiple obstacles, each affecting the drone
    obstacles = [Obstacle(0, 6, 3), Obstacle(6, 4, 2), Obstacle(10, 2, 1)]
    x = 3
    y = 2
    expected_result = 120686059.8
    actual_result = gaussFunction(x, y, obstacles, 255000000)
    assert math.isclose(actual_result, expected_result,
                        abs_tol=1e-6), f"Test Case 5 failed. Actual: {actual_result}, Expected: {expected_result}"
    print("Test Case 5 passed successfully.")

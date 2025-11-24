import numpy as np
import heapq

class AStarPlanner:
    def __init__(self, grid, resolution, origin):
        self.grid = grid
        self.resolution = resolution
        self.origin = origin
        self.height, self.width = grid.shape

    def plan(self, start, goal):
        """
        Plans a path from start to goal using the A* algorithm.
        start and goal are in world coordinates.
        Returns a list of waypoints in world coordinates.
        """
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        if not self._is_valid(start_grid) or not self._is_valid(goal_grid):
            print("Start or goal is not valid.")
            return []

        open_list = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal_grid:
                return self._reconstruct_path(came_from, current)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self._is_valid(neighbor):
                    continue

                tentative_g_score = g_score[current] + self._heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return []

    def _reconstruct_path(self, came_from, current):
        """Reconstructs the path from the came_from dictionary."""
        path = [self._grid_to_world(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self._grid_to_world(current))
        path.reverse()
        return path


    def _world_to_grid(self, point):
        """Converts a point from world coordinates to grid coordinates."""
        x, y = point
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (grid_y, grid_x)

    def _grid_to_world(self, point):
        """Converts a point from grid coordinates to world coordinates."""
        row, col = point
        x = (col + 0.5) * self.resolution + self.origin[0]
        y = (row + 0.5) * self.resolution + self.origin[1]
        return (x, y)

    def _is_valid(self, point):
        """Checks if a point is valid (within the grid and not an obstacle)."""
        row, col = point
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        if self.grid[row, col] == 100:
            return False
        return True

    def _heuristic(self, a, b):
        """Calculates the heuristic distance between two points."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

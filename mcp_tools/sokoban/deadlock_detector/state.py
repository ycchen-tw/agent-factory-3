"""
Reference: https://github.com/shiandrew/AI-Driven-Sokoban-Solver/blob/main/state.py
"""
from typing import List, Tuple, Set
import numpy as np

class SokobanState:
    # Constants for the game elements
    WALL = '#'
    BOX = '$'
    PLAYER = '@'
    GOAL = '.'
    FLOOR = ' '
    BOX_ON_GOAL = '*'
    PLAYER_ON_GOAL = '+'

    def __init__(self, grid_data: List[str]):
        """
        Initialize the Sokoban state from a grid representation.
        
        Args:
            grid_data: List of strings representing the grid, where each string is a row
        """
        # Normalize the grid to handle irregular shapes
        self.grid = self._normalize_grid(grid_data)
        self.height = len(self.grid)
        self.width = max(len(row) for row in self.grid) if self.grid else 0
        self.player_pos = self._find_player()
        self.boxes = self._find_boxes()
        self.goals = self._find_goals()
    
    def _normalize_grid(self, grid_data: List[str]) -> List[str]:
        """
        Normalize the grid to ensure all rows have the same length.
        Pads shorter rows with spaces and removes completely empty rows.
        
        Args:
            grid_data: List of strings representing the grid
            
        Returns:
            Normalized grid with consistent row lengths
        """
        # Remove completely empty rows at the beginning and end
        while grid_data and not grid_data[0].strip():
            grid_data = grid_data[1:]
        while grid_data and not grid_data[-1].strip():
            grid_data = grid_data[:-1]
        
        if not grid_data:
            raise ValueError("Empty grid provided")
        
        # Find the maximum width
        max_width = max(len(row) for row in grid_data)
        
        # Pad all rows to the same width
        normalized = []
        for row in grid_data:
            # Pad with spaces to reach max_width
            padded_row = row.ljust(max_width)
            normalized.append(padded_row)
        
        return normalized
        
    def _find_player(self) -> Tuple[int, int]:
        """Find the player's position in the grid."""
        for y in range(self.height):
            for x in range(len(self.grid[y])):
                if self.grid[y][x] in [self.PLAYER, self.PLAYER_ON_GOAL]:
                    return (y, x)
        raise ValueError("No player found in grid")

    def _find_boxes(self) -> Set[Tuple[int, int]]:
        """Find all boxes in the grid."""
        boxes = set()
        for y in range(self.height):
            for x in range(len(self.grid[y])):
                if self.grid[y][x] in [self.BOX, self.BOX_ON_GOAL]:
                    boxes.add((y, x))
        return boxes

    def _find_goals(self) -> Set[Tuple[int, int]]:
        """Find all goal positions in the grid."""
        goals = set()
        for y in range(self.height):
            for x in range(len(self.grid[y])):
                if self.grid[y][x] in [self.GOAL, self.BOX_ON_GOAL, self.PLAYER_ON_GOAL]:
                    goals.add((y, x))
        return goals

    def is_goal_state(self) -> bool:
        """Check if all boxes are on goals."""
        return all(box in self.goals for box in self.boxes)

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves for the player."""
        valid_moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        for dy, dx in directions:
            new_y, new_x = self.player_pos[0] + dy, self.player_pos[1] + dx
            
            # Check if the new position is within bounds
            if not (0 <= new_y < self.height and 0 <= new_x < self.width):
                continue
                
            # Check if the new position is a wall
            if new_x >= len(self.grid[new_y]) or self.grid[new_y][new_x] == self.WALL:
                continue
                
            # Check if the new position has a box
            if (new_y, new_x) in self.boxes:
                # Check if we can push the box
                box_new_y, box_new_x = new_y + dy, new_x + dx
                if not (0 <= box_new_y < self.height and 0 <= box_new_x < self.width):
                    continue
                if (box_new_x >= len(self.grid[box_new_y]) or 
                    self.grid[box_new_y][box_new_x] == self.WALL):
                    continue
                if (box_new_y, box_new_x) in self.boxes:
                    continue
                    
            valid_moves.append((dy, dx))
            
        return valid_moves

    def move(self, direction: Tuple[int, int]) -> 'SokobanState':
        """
        Create a new state by moving the player in the given direction.
        
        Args:
            direction: Tuple (dy, dx) representing the direction to move
            
        Returns:
            New SokobanState after the move
        """
        dy, dx = direction
        new_y, new_x = self.player_pos[0] + dy, self.player_pos[1] + dx
        
        # Create a new grid representation
        new_grid = [list(row) for row in self.grid]
        
        # Update player position
        old_y, old_x = self.player_pos
        if (old_y, old_x) in self.goals:
            new_grid[old_y][old_x] = self.GOAL
        else:
            new_grid[old_y][old_x] = self.FLOOR
            
        # Check if we're moving a box
        if (new_y, new_x) in self.boxes:
            box_new_y, box_new_x = new_y + dy, new_x + dx
            # Move the box
            if (box_new_y, box_new_x) in self.goals:
                new_grid[box_new_y][box_new_x] = self.BOX_ON_GOAL
            else:
                new_grid[box_new_y][box_new_x] = self.BOX
                
        # Update player position
        if (new_y, new_x) in self.goals:
            new_grid[new_y][new_x] = self.PLAYER_ON_GOAL
        else:
            new_grid[new_y][new_x] = self.PLAYER
            
        return SokobanState([''.join(row) for row in new_grid])

    def __eq__(self, other: 'SokobanState') -> bool:
        """Check if two states are equal."""
        if not isinstance(other, SokobanState):
            return False
        return (self.player_pos == other.player_pos and 
                self.boxes == other.boxes)

    def __hash__(self) -> int:
        """Create a hash of the state for use in sets/dictionaries."""
        return hash((self.player_pos, frozenset(self.boxes)))

    def __str__(self) -> str:
        """String representation of the state."""
        return '\n'.join(self.grid)
# Saving Heuristic functions:
import numpy as np
# Chosen heuristic
def heuristic(self) -> int:
    distances = []  # List to store distances to extra energy cells
    targets = []  # List to store distances to target cells
    curr_pos = self.current_position  # Current position (row, col)

    # Calculate distances to extra energy cells and target cells
    for row in range(self.board_row_size):
        for col in range(self.board_col_size):
            # Check if the cell has extra energy and calculate the distance
            if len(self.board[row][col]) > 1 and self.board[row][col][1] in self.extra_energy.keys():
                distance = np.absolute((curr_pos[0] - row) + (curr_pos[1] - col))
                distances.append(distance)

            # Check if the cell is a target and calculate the distance
            if 'T' in self.board[row][col]:
                target = np.absolute((curr_pos[0] - row) + (curr_pos[1] - col))
                targets.append(target)

    # Find the minimum distances to extra energy and target cells
    min_distance = 0
    min_target = 0
    if len(distances) != 0:
        min_distance = np.min(distances)
    if len(targets) != 0:
        min_target = np.min(targets)

    # Combine distances with a heuristic function
    heuristic = -min_distance - min_target
    return heuristic

# Other heuristic:

# def heuristic(self) -> float:
#
#     min_distance = float('inf')  # Initialize minimum distance to infinity
#
#     max_energy = 0  # Initialize maximum energy gain to 0
#
#     min_energy_distance = float('inf')
#
#     curr_pos = self.current_position  # Get the current position
#
#
#
#     for i in range(self.board_row_size):
#
#         for j in range(self.board_col_size):
#
#             if 'T' in self.board[i][j]:  # If the cell contains a target
#
#                 # Calculate the Manhattan distance to the target
#
#                 distance = abs(curr_pos[0] - i) + abs(curr_pos[1] - j)
#
#                 min_distance = min(min_distance, distance)  # Update the minimum distance
#
#
#
#             if len(self.board[i][j]) > 1 and self.board[i][j][1] in self.extra_energy.keys():
#
#                 # If the cell contains an energy point, update the maximum energy gain
#
#                 max_energy = max(max_energy, self.extra_energy[self.board[i][j][1]])
#
#                 min_energy_distance = min(min_energy_distance, abs(curr_pos[0] - i) + abs(curr_pos[1] - j))
#
#
#
#     # Combine the distance and energy factors to form the heuristic value
#
#     heuristic_value = -(0.3 * min_distance + 0.4 * max_energy + 0.3 * min_energy_distance)
#
#     return heuristic_value
#
########
# def manhattan_distance(self,currposition, target):
#     x1, y1 = currposition
#     x2, y2 = target
#     return abs(x1 - x2) + abs(y1 - y2)
#
# def heuristic(self,cell ) :
#     targets = self.Targets_founder()  # Get the remaining target positions
#     for target in targets :
#         if target.is_remaining_target():
#             target_distances = [self.manhattan_distance(cell, target) for target in targets]
#             min_distance = min(target_distances)
#             return min_distance
#

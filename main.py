from collections import deque
import timeit
import heapq
import numpy as np


# TODO: Posistion class

class Board:
    def __init__(self, board, row_size, col_size, current_position: (int, int), energy: int):
        self.board = board
        self.current_position = current_position

        self.board_row_size = row_size
        self.board_col_size = col_size

        self.path_to_parent = []

        self.moves = ['L', 'D', 'R', 'U']
        self.extra_energy = {'C': 10, 'B': 5, 'I': 12}
        self.other_notations = {'R', 'T'}
        self.energy = energy
        # self.cost = cost

    def add_path(self, move):
        self.path_to_parent.append(move)

    def path_validity_check(self, dest: (int, int)) -> bool:
        d_row, d_col = dest
        path_is_valid = True

        if d_row > self.board_row_size - 1 or d_col > self.board_col_size - 1 or d_row < 0 or d_col < 0:
            path_is_valid = False

        elif self.board[d_row][d_col] == 'X':
            path_is_valid = False

        return path_is_valid

    def move_validity(self, current_state: (int, int), op_code: str) -> (int, int):
        row, col = current_state
        dest_state = (-1, -1)

        if op_code == 'U':
            dest_state = (row - 1, col)

        elif op_code == 'D':
            dest_state = (row + 1, col)

        elif op_code == 'R':
            dest_state = (row, col + 1)

        elif op_code == 'L':
            dest_state = (row, col - 1)

        if self.path_validity_check(dest_state):
            return dest_state

        return (-1, -1)

    def available_moves(self, current_state: (int, int), cost=False) -> list:
        valid_moves = []
        for _move in self.moves:
            neighbor = self.move_validity(current_state, _move)
            if neighbor != (-1, -1):
                if cost:
                    value = self.calculate_energy(neighbor)
                    valid_moves.append((_move, value))
                else:
                    valid_moves.append(_move)

            # min_value = map(lambda x, y: x[1] < y[1], va)

        return valid_moves

    def update_target(self, position: (int, int)) -> None:
        row_index, col_index = position
        self.board[row_index][col_index] = self.board[row_index][col_index][0]

    def calculate_energy(self, position: (int, int)) -> None:
        row_index, col_index = position

        extra_energy_cell: str = self.board[row_index][col_index]

        if len(extra_energy_cell) > 1:

            if extra_energy_cell[1] in self.extra_energy.keys():

                # remove extra energy from board
                self.board[row_index][col_index] = extra_energy_cell[0]

                # update energy
                plus_energy = self.extra_energy.get(extra_energy_cell[1])

                minus_energy = int(extra_energy_cell[0])
                energy = plus_energy - minus_energy

                return energy

            elif extra_energy_cell[1] in self.other_notations:
                return -int(extra_energy_cell[0])

        cell_energy_consume = int(self.board[row_index][col_index])
        return -cell_energy_consume

    def calculate_path_energy(self, moves: list, current_position: (int, int)) -> int:
        total_energy = 0
        for move in moves:
            position = self.move_validity(current_position, move)
            energy = self.calculate_energy(position)
            total_energy += energy
            current_position = position

        return total_energy

    def __str__(self) -> str:
        row, col = self.current_position
        return self.board[row][col]

    def __lt__(self, other):
        return self.energy < other.energy

    def is_remaining_target(self) -> bool:
        is_T = False
        for i in self.board:
            for j in i:
                if 'T' in j:
                    is_T = True

        return is_T

    def heuristic(self) -> int:
        # Define a high value for initial minimum distance
        min_distance = float('inf')
        # Get the current position
        curr_pos = self.current_position

        for i in range(self.board_row_size):
            for j in range(self.board_col_size):
                # Check every cell in the board if there's a remaining target
                if 'T' in self.board[i][j]:
                    # If there's a target, calculate Manhattan distance
                    distance = abs(curr_pos[0] - i) + abs(curr_pos[1] - j)
                    # Update the minimum distance
                    min_distance = min(min_distance, distance)
        return min_distance

    def heuristic2(self) -> int:
        # List to store distances
        distances = []

        # Get the current position
        curr_pos = self.current_position

        for i in range(self.board_row_size):
            for j in range(self.board_col_size):
                # Check every cell in the board if there's an energy point
                if len(self.board[i][j]) > 1 and self.board[i][j][1] in self.extra_energy.keys():
                    # Calculate the Euclidean distance from current position to energy point
                    distance = np.sqrt((curr_pos[0] - i) ** 2 + (curr_pos[1] - j) ** 2)
                    distances.append(distance)

        # Return infinity if no energy point is found (distances list is empty)
        if len(distances) == 0:
            return np.inf

        # Calculate the average of the distances
        avg_distance = np.mean(distances)

        return avg_distance


class Tree:

    def __init__(self, matrix):
        self.tree: dict[Board: list[Board]] = dict()
        self.Number_Of_Target_Found = 0
        self.Number_Of_Target = len(self.Targets_founder(matrix))

    def all_targets_found(self, path, targets, initial_position):
        x, y = initial_position
        targets_found = []
        for move in path:
            if move == 'R':
                y += 1
            elif move == 'L':
                y -= 1
            elif move == 'U':
                x -= 1
            elif move == 'D':
                x += 1
            current_position = (x, y)
            if current_position in targets:
                targets_found.append(current_position)
        return set(targets) == set(targets_found)

    def get_visited_positions(self, initial_position, directions):
        visited_positions = [initial_position]
        current_position = initial_position

        for direction in directions:
            if direction == "R":
                current_position = (current_position[0], current_position[1] + 1)
            elif direction == "L":
                current_position = (current_position[0], current_position[1] - 1)
            elif direction == "U":
                current_position = (current_position[0] - 1, current_position[1])
            elif direction == "D":
                current_position = (current_position[0] + 1, current_position[1])

            visited_positions.append(current_position)

        return visited_positions

    def Targets_founder(self, Matrix):
        # Create a dictionary to map positions to the number 0
        positions_to_numbers = []

        # Iterate over the matrix and find positions of 'T'
        for i, row in enumerate(Matrix):
            for j, element in enumerate(row):
                if 'T' in element:
                    positions_to_numbers.append((i, j))
        return positions_to_numbers

    def add_edge(self, root: Board, dest: Board):
        if root in self.tree:
            self.tree.get(root).append(dest)

        else:
            self.tree[root] = [dest]

    def dfs(self, root_node: Board) -> Board | None:
        visited = {}
        stack = OrderedSet()
        stack.add(root_node)

        while stack and self.Number_Of_Target_Found < self.Number_Of_Target:
            current_node = stack.pop()
            visited[current_node.current_position] = current_node.energy
            curr_pos = current_node.current_position  # (row, col)
            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                current_node.update_target(current_node.current_position)

                # print(current_node.path_to_parent)
                self.Number_Of_Target_Found += 1

                if self.Number_Of_Target_Found >= self.Number_Of_Target:
                    return current_node

                temp = self.dfs(current_node)
                current_node.path_to_parent = temp.path_to_parent
                current_node.energy = temp.energy

            moves = current_node.available_moves(curr_pos)  # successor function
            # print(moves)

            for move in moves:
                new_position = current_node.move_validity(curr_pos, move)
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    current_node.energy + current_node.calculate_energy(new_position)
                )

                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)

                child_node.add_path(move)

                if child_node.current_position not in visited:
                    stack.add(child_node)
                    visited[child_node.current_position] = child_node.energy
                elif child_node.current_position in visited:
                    if child_node.energy >= visited[child_node.current_position]:
                        stack.add(child_node)
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

        return current_node

    def bfs(self, root_node: Board) -> Board | None:
        queue = deque([root_node])
        visited = {}

        while queue and self.Number_Of_Target_Found < self.Number_Of_Target:
            current_node = queue.popleft()
            visited[current_node.current_position] = current_node.energy
            curr_pos = current_node.current_position  # (row, col)
            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                current_node.update_target(current_node.current_position)

                self.Number_Of_Target_Found += 1
                if self.Number_Of_Target_Found >= self.Number_Of_Target:
                    return current_node

                temp = self.bfs(current_node)
                current_node.path_to_parent = temp.path_to_parent
                current_node.energy = temp.energy

            moves = current_node.available_moves(curr_pos)  # successor function
            for move in moves:

                new_position = current_node.move_validity(curr_pos, move)

                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    current_node.energy + current_node.calculate_energy(new_position)
                )

                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)

                child_node.add_path(move)

                if child_node.current_position not in visited:
                    queue.append(child_node)
                    visited[child_node.current_position] = child_node.energy
                elif child_node.current_position in visited:
                    if child_node.energy >= visited[child_node.current_position]:
                        queue.append(child_node)
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

        return current_node

    def ucs(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = {}

        while priority_queue and self.Number_Of_Target_Found < self.Number_Of_Target:
            _, current_node = heapq.heappop(priority_queue)
            visited[current_node.current_position] = current_node.energy
            curr_pos = current_node.current_position
            moves = current_node.available_moves(curr_pos, cost=True)  # successor

            for move, move_cost in moves:
                total_cost = current_node.energy + move_cost
                new_position = current_node.move_validity(curr_pos, move)
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    total_cost
                    # current_node.energy + current_node.calculate_energy(new_position)
                )

                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)

                child_node.add_path(move)

                if child_node.current_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                    visited[child_node.current_position] = child_node.energy
                elif child_node.current_position in visited:
                    if child_node.energy >= visited[child_node.current_position]:
                        heapq.heappush(priority_queue, (-total_cost, child_node))
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    self.Number_Of_Target_Found += 1
                    if self.Number_Of_Target_Found >= self.Number_Of_Target:
                        return current_node

                    temp = self.ucs(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

        return current_node

    def ucs(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = {}

        while priority_queue and self.Number_Of_Target_Found < self.Number_Of_Target:
            _, current_node = heapq.heappop(priority_queue)
            visited[current_node.current_position] = current_node.energy
            curr_pos = current_node.current_position
            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                current_node.update_target(current_node.current_position)

                self.Number_Of_Target_Found += 1
                if self.Number_Of_Target_Found >= self.Number_Of_Target:
                    return current_node

                temp = self.ucs(current_node)
                current_node.path_to_parent = temp.path_to_parent
                current_node.energy = temp.energy

            moves = current_node.available_moves(curr_pos, cost=True)  # successor
            for move, move_cost in moves:
                total_cost = current_node.energy + move_cost
                new_position = current_node.move_validity(curr_pos, move)
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    total_cost
                    # current_node.energy + current_node.calculate_energy(new_position)
                )

                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)

                child_node.add_path(move)

                if child_node.current_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                    visited[child_node.current_position] = child_node.energy
                elif child_node.current_position in visited:
                    if child_node.energy >= visited[child_node.current_position]:
                        heapq.heappush(priority_queue, (-total_cost, child_node))
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

        return current_node

    def astar(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = {}

        while priority_queue and self.Number_Of_Target_Found < self.Number_Of_Target:
            _, current_node = heapq.heappop(priority_queue)
            visited[current_node.current_position] = current_node.energy
            curr_pos = current_node.current_position
            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                current_node.update_target(current_node.current_position)

                self.Number_Of_Target_Found += 1
                if self.Number_Of_Target_Found >= self.Number_Of_Target:
                    return current_node

                temp = self.astar(current_node)
                current_node.path_to_parent = temp.path_to_parent
                current_node.energy = temp.energy

            moves = current_node.available_moves(curr_pos, cost=True)  # successor

            for move, move_cost in moves:
                total_cost = current_node.energy + move_cost + current_node.heuristic2()
                new_position = current_node.move_validity(curr_pos, move)
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    total_cost
                    # current_node.energy + current_node.calculate_energy(new_position)
                )

                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)
                child_node.add_path(move)
                child_node.energy = current_node.energy + move_cost

                if child_node.current_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                    visited[child_node.current_position] = child_node.energy
                elif child_node.current_position in visited:
                    if child_node.energy >= visited[child_node.current_position]:
                        heapq.heappush(priority_queue, (-total_cost, child_node))
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

        return current_node

    def best_first_search(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = {}

        while priority_queue and self.Number_Of_Target_Found < self.Number_Of_Target:
            _, current_node = heapq.heappop(priority_queue)
            visited[current_node.current_position] = current_node.energy
            curr_pos = current_node.current_position
            moves = current_node.available_moves(curr_pos, cost=True)  # successor

            for move, move_cost in moves:
                total_cost = current_node.energy + current_node.heuristic2()
                new_position = current_node.move_validity(curr_pos, move)
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    total_cost
                    # current_node.energy + current_node.calculate_energy(new_position)
                )

                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)
                child_node.add_path(move)
                child_node.energy = current_node.energy + move_cost

                if child_node.current_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                    visited[child_node.current_position] = child_node.energy
                elif child_node.current_position in visited:
                    if child_node.energy >= visited[child_node.current_position]:
                        heapq.heappush(priority_queue, (-total_cost, child_node))
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    self.Number_Of_Target_Found += 1
                    if self.Number_Of_Target_Found >= self.Number_Of_Target:
                        return current_node

                    temp = self.astar(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

        return current_node


class OrderedSet:
    def __init__(self):
        self.list = []

    def add(self, element):
        if element not in self.list:
            self.list.append(element)

    def pop(self):
        return self.list.pop()


Matrix_main = [
    ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
    ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
    ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
    ['1I', '1T', '1', '2', '2', '2T', '2', '1', '1', '1']
]
initial_energy = 500
initial_position = (0, 0)

####### BFS #######
matrix = [
    ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
    ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
    ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
    ['1I', '1T', '1', '2', '2', '2T', '2', '1', '1', '1']
]

print("BFS = ")
board = Board(matrix, 6, 10, initial_position, initial_energy)
tree = Tree(matrix)
start = timeit.default_timer()
result = tree.bfs(board)
if tree.all_targets_found(result.path_to_parent, tree.Targets_founder(Matrix_main), initial_position):
    print(result.path_to_parent)
    print(result.energy)
else:
    print("there is no route")

end = timeit.default_timer()
print("time elapsed: {:f}s".format(end - start))
####### DFS #######
matrix = [
    ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
    ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
    ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
    ['1I', '1T', '1', '2', '2', '2T', '2', '1', '1', '1']
]

print("DFS = ")
board2 = Board(matrix, 6, 10, initial_position, initial_energy)
tree2 = Tree(matrix)
start = timeit.default_timer()
try:
    result2 = tree2.dfs(board2)
    print(result2.path_to_parent)
    print(result2.energy)
except IndexError:
    print("there is no route")
end = timeit.default_timer()
print("time elapsed: {:f}s".format(end - start))

# ####### IDS ########
# matrix = [
#     ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
#     ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
#     ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
#     ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
#     ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
#     ['1I', '1T', '1', '2', '2', '2T', '2', '1', '1', '1']
# ]
#
# print("IDS = ")
# board_ids = Board(matrix, 6, 10, initial_position, initial_energy)
# tree_ids = Tree(matrix)
# max_depth_limit = 50000
# start = timeit.default_timer()
# result_ids = tree_ids.ids(board_ids, max_depth_limit)
# if tree_ids.all_targets_found(result_ids.path_to_parent, tree_ids.Targets_founder(Matrix_main), initial_position):
#     print(result_ids.path_to_parent)
#     print(result_ids.energy)
# else:
#     print("there is no route")
# end = timeit.default_timer()
# print("time elapsed: {:f}s".format(end - start))

####### UCS #######
matrix = [
    ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
    ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
    ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
    ['1I', '1T', '1', '2', '2', '2T', '2', '1', '1', '1']
]

print("UCS = ")
board3 = Board(matrix, 6, 10, initial_position, initial_energy)
tree3 = Tree(matrix)
start = timeit.default_timer()
result3 = tree3.ucs(board3)
if tree3.all_targets_found(result3.path_to_parent, tree3.Targets_founder(Matrix_main), initial_position):
    print(result3.path_to_parent)
    print(result3.energy)
else:
    print("there is no route")
end = timeit.default_timer()
print("time elapsed: {:f}s".format(end - start))

####### َA* #######
matrix = [
    ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
    ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
    ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
    ['1I', '1T', '1', '2', '2', '2T', '2', '1', '1', '1']
]

print("َA* = ")
board4 = Board(matrix, 6, 10, initial_position, initial_energy)
tree4 = Tree(matrix)
start = timeit.default_timer()
result4 = tree4.astar(board4)
if tree4.all_targets_found(result4.path_to_parent, tree4.Targets_founder(Matrix_main), initial_position):
    print(result4.path_to_parent)
    print(result4.energy)
else:
    print("there is no route")
end = timeit.default_timer()
print("time elapsed: {:f}s".format(end - start))

###### Best First Search #####
matrix = [
    ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
    ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
    ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
    ['1I', '1T', '1', '2', '2', '2T', '2', '1', '1', '1']
]

print("Best First Search* = ")
board5 = Board(matrix, 6, 10, initial_position, initial_energy)
tree5 = Tree(matrix)
start = timeit.default_timer()
result5 = tree5.best_first_search(board5)
if tree5.all_targets_found(result5.path_to_parent, tree5.Targets_founder(Matrix_main), initial_position):
    print(result5.path_to_parent)
    print(result5.energy)
else:
    print("there is no route")
end = timeit.default_timer()
print("time elapsed: {:f}s".format(end - start))

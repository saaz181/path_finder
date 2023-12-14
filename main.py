from collections import deque
import time
import heapq


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
        # Define a high value for initial minimum distance
        min_distance = float('inf')
        # Define a low value for initial maximum energy
        max_energy = float('-inf')
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

                # Check every cell in the board if there's an energy point
                if len(self.board[i][j]) > 1 and self.board[i][j][1] in self.extra_energy.keys():
                    # If there's an energy point, update the maximum energy
                    max_energy = max(max_energy, self.extra_energy[self.board[i][j][1]])

        # Balance both the distance to target and energy points
        return min_distance - max_energy

class Tree:

    def __init__(self, matrix):
        self.tree: dict[Board: list[Board]] = dict()
        self.Number_Of_Target_Found = 0
        self.Number_Of_Target = len(self.Targets_founder(matrix))

    def All_Target_found(self, Path, Target, current_node):
        for element in Target:
            if element != current_node:
                if element not in Path:
                    return False
        return True

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
        visited = set()
        visited.add((root_node.current_position))
        stack = OrderedSet()
        stack.add(root_node)

        while stack and self.Number_Of_Target_Found < self.Number_Of_Target:
            current_node = stack.pop()
            # print(current_node.current_position)
            curr_pos = current_node.current_position  # (row, col)

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

                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    # print(current_node.path_to_parent)
                    self.Number_Of_Target_Found += 1

                    if self.Number_Of_Target_Found >= self.Number_Of_Target:
                        return current_node

                    temp = self.dfs(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

                if child_node.current_position not in visited:
                    visited.add(child_node.current_position)
                    stack.add(child_node)
                else:
                    continue

        return current_node

    def bfs(self, root_node: Board) -> Board | None:
        queue = deque([root_node])
        visited = set()
        visited.add(root_node)

        while queue and self.Number_Of_Target_Found < self.Number_Of_Target:
            current_node = queue.popleft()

            curr_pos = current_node.current_position  # (row, col)

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

                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    self.Number_Of_Target_Found += 1
                    if self.Number_Of_Target_Found >= self.Number_Of_Target:
                        return current_node

                    temp = self.bfs(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

                if child_node not in visited:
                    queue.append(child_node)
                    visited.add(child_node)
                else:
                    continue

        return current_node

    def ucs(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = set()

        while priority_queue and self.Number_Of_Target_Found < self.Number_Of_Target:
            _, current_node = heapq.heappop(priority_queue)
            visited.add(current_node.current_position)
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

                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    self.Number_Of_Target_Found += 1
                    if self.Number_Of_Target_Found >= self.Number_Of_Target:
                        return current_node

                    temp = self.ucs(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

                if new_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                else:
                    continue
        return current_node

    def astar(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = set()

        while priority_queue and self.Number_Of_Target_Found < self.Number_Of_Target:
            _, current_node = heapq.heappop(priority_queue)
            visited.add(current_node.current_position)
            curr_pos = current_node.current_position
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

                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    self.Number_Of_Target_Found += 1
                    if self.Number_Of_Target_Found >= self.Number_Of_Target:
                        return current_node

                    temp = self.ucs(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

                if new_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                else:
                    continue
        return current_node



class OrderedSet:
    def __init__(self):
        self.list = []

    def add(self, element):
        if element not in self.list:
            self.list.append(element)

    def pop(self):
        return self.list.pop()


initial_energy = 500
initial_position = (0, 0)

####### BFS #######
matrix = [
    ['1R', '1', '1', '5', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '3', '5', '5', '5', '4', '5', 'X'],
    ['5', '1I', '1', '6', '2', '2', '2', '1', '1', '1T'],
    ['X', 'X', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', '1', 'X', 'X', '50', '2', '1C', '1', 'X'],
    ['1', '1', '1', '2', '2', '2T', '2', '1', '1', '1']
]
print("BFS = ")
board = Board(matrix, 6, 10, initial_position, initial_energy)
tree = Tree(matrix)
result = tree.bfs(board)
print(result.path_to_parent)
print(result.energy)
####### DFS #######
matrix = [
    ['1R', '1', '1', '5', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '3', '5', '5', '5', '4', '5', 'X'],
    ['5', '1I', '1', '6', '2', '2', '2', '1', '1', '1T'],
    ['X', 'X', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', '1', 'X', 'X', '50', '2', '1C', '1', 'X'],
    ['1', '1', '1', '2', '2', '2T', '2', '1', '1', '1']
]

print("DFS = ")
board2 = Board(matrix, 6, 10, initial_position, initial_energy)
tree2 = Tree(matrix)
result2 = tree2.dfs(board2)
print(result2.path_to_parent)
print(result2.energy)

####### UCS #######
matrix = [
    ['1R', '1', '1', '5', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '3', '5', '5', '5', '4', '5', 'X'],
    ['5', '1I', '1', '6', '2', '2', '2', '1', '1', '1T'],
    ['X', 'X', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', '1', 'X', 'X', '50', '2', '1C', '1', 'X'],
    ['1', '1', '1', '2', '2', '2T', '2', '1', '1', '1']
]
print("UCS = ")
board3 = Board(matrix, 6, 10, initial_position, initial_energy)
tree3 = Tree(matrix)
start = time.time()
result3 = tree3.ucs(board3)
print(result3.path_to_parent)
print(result3.energy)
print("time elapsed: {:.2f}s".format(time.time() - start))

####### َA* #######
matrix = [
    ['1R', '1', '1', '5', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '3', '5', '5', '5', '4', '5', 'X'],
    ['5', '1I', '1', '6', '2', '2', '2', '1', '1', '1T'],
    ['X', 'X', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', '1', 'X', 'X', '50', '2', '1C', '1', 'X'],
    ['1', '1', '1', '2', '2', '2T', '2', '1', '1', '1']
]
print("َA* = ")
board4 = Board(matrix, 6, 10, initial_position, initial_energy)
tree4 = Tree(matrix)
start = time.time()
result = tree4.astar(board4)
print(result.path_to_parent)
print(result.energy)
print("time elapsed: {:.2f}s".format(time.time() - start))
from collections import deque
import heapq
import numpy as np


class OrderedSet:
    def __init__(self):
        self.list = []

    def add(self, element):
        if element not in self.list:
            self.list.append(element)

    def pop(self):
        return self.list.pop()



class Board:
    def __init__(self, board: list[list], row_size: int, col_size: int, current_position: (int, int), energy: int):
        self.board = board
        self.current_position = current_position

        self.board_row_size = row_size
        self.board_col_size = col_size

        self.path_to_parent = []

        self.moves = ['L', 'D', 'R', 'U']
        self.extra_energy = {'C': 10, 'B': 5, 'I': 12}
        self.other_notations = {'R', 'T'}
        self.energy = energy

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
        distances = []
        targets = []
        curr_pos = self.current_position
        #print(curr_pos)
        for row in range(self.board_row_size):
            for col in range(self.board_col_size):
                if len(self.board[row][col]) > 1 and self.board[row][col][1] in self.extra_energy.keys():
                    distance = np.absolute((curr_pos[0] - row)  + (curr_pos[1] - col))
                    distances.append(distance)
                if 'T' in self.board[row][col]:
                    target = np.absolute((curr_pos[0] - row) + (curr_pos[1] - col))
                    targets.append(target)

        min_distance = 0
        min_target = 0
        if len(distances) != 0:
            min_distance = np.min(distances)
        if len(targets) != 0:
            min_target = np.min(targets)

        heuristic = -min_distance - min_target
        return heuristic
    

class Tree:
    def __init__(self, matrix):
        self.tree: dict[Board: list[Board]] = dict()
        self.found_targets = 0
        self.targets = len(self.target_numerator(matrix))

    def target_numerator(self, matrix):
        # Create a dictionary to map positions to the number 0
        positions_to_numbers = []

        # Iterate over the matrix and find positions of 'T'
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                if 'T' in element:
                    positions_to_numbers.append((i, j))
        return positions_to_numbers

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

    def dfs(self, root_node: Board) -> Board | None:
        visited = {}
        stack = OrderedSet()
        stack.add(root_node)

        while stack and self.found_targets < self.targets:
            current_node = stack.pop()
            
            visited[current_node.current_position] = current_node.energy
            
            curr_pos = current_node.current_position  # (row, col)
            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                current_node.update_target(current_node.current_position)

                self.found_targets += 1

                if self.found_targets >= self.targets:
                    return current_node

                temp = self.dfs(current_node)
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

        while queue and self.found_targets < self.targets:
            current_node = queue.popleft()
            visited[current_node.current_position] = current_node.energy
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

                    self.found_targets += 1
                    if self.found_targets >= self.targets:
                        return current_node

                    temp = self.bfs(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

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

    def ids(self, root_node: Board, max_depth: int) -> Board | None:
        depth_limit = 1
        visited = dict()
        while depth_limit < max_depth:
            result = self._recursive_dls(root_node, depth_limit, visited)
            if result is not None:
                return result
            depth_limit += 1

    def _recursive_dls(self, current_node: Board, depth_limit: int, visited: dict) -> Board | None:
        if depth_limit == 0:
            return None  # Reached depth limit, no solution found at this level

        curr_pos = current_node.current_position
        moves = current_node.available_moves(curr_pos)  # successor function

        
        if current_node.current_position not in visited:
            visited[current_node.current_position] = current_node.energy
                
        elif current_node.current_position in visited:
            if current_node.energy >= visited[current_node.current_position]:
                visited[current_node.current_position] = current_node.energy
                self._recursive_dls(current_node, depth_limit - 1, visited)

        

        for move in moves:
            new_position: (int, int) = current_node.move_validity(curr_pos, move)
            child_energy: int = current_node.energy + current_node.calculate_energy(new_position)
            
            child_node = Board(
                current_node.board,
                current_node.board_row_size,
                current_node.board_col_size,
                new_position,
                child_energy
            )

            for _move in current_node.path_to_parent:
                child_node.add_path(_move)

            child_node.add_path(move)

            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                current_node.update_target(current_node.current_position)

                return current_node

            result = self._recursive_dls(child_node, depth_limit - 1, visited)

            if result is not None:
                return result

        return None  # No solution found at this level

    def ucs(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = {}

        while priority_queue and self.found_targets < self.targets:
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

                    self.found_targets += 1
                    if self.found_targets >= self.targets:
                        return current_node

                    temp = self.ucs(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

        return current_node

    def astar(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = {}

        while priority_queue and self.found_targets < self.targets:
            _, current_node = heapq.heappop(priority_queue)
            
            visited[current_node.current_position] = current_node.energy
            
            curr_pos = current_node.current_position
            moves = current_node.available_moves(curr_pos, cost=True)  # successor

            for move, move_cost in moves:
                total_cost = current_node.energy + move_cost + current_node.heuristic()
                new_position = current_node.move_validity(curr_pos, move)
                
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    total_cost
                )

                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)
                child_node.add_path(move)

                # child_node.energy = current_node.energy + move_cost

                if child_node.current_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                    visited[child_node.current_position] = child_node.energy
                
                elif child_node.current_position in visited:
                
                    if child_node.energy > visited[child_node.current_position]:
                        heapq.heappush(priority_queue, (-total_cost, child_node))
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    self.found_targets += 1
                    if self.found_targets >= self.targets:
                        return current_node

                    temp = self.astar(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

        return current_node

    def best_first_search(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = {}

        while priority_queue and self.found_targets < self.targets:
            _, current_node = heapq.heappop(priority_queue)
            
            visited[current_node.current_position] = current_node.energy
            curr_pos = current_node.current_position
            moves = current_node.available_moves(curr_pos, cost=True)  # successor

            for move, move_cost in moves:
                total_cost = current_node.energy + current_node.heuristic()
                new_position = current_node.move_validity(curr_pos, move)
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    total_cost
                )

                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)
                child_node.add_path(move)
                
                # child_node.energy = current_node.energy + move_cost

                if child_node.current_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                    visited[child_node.current_position] = child_node.energy
                
                elif child_node.current_position in visited:
                
                    if child_node.energy > visited[child_node.current_position]:
                        heapq.heappush(priority_queue, (-total_cost, child_node))
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    self.found_targets += 1
                    if self.found_targets >= self.targets:
                        return current_node

                    temp = self.best_first_search(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

        return current_node

from typing import Optional
from collections import deque


# TODO: Position class


class Board:
    def __init__(self, board, row_size, col_size, current_position: (int, int), energy: int):
        self.board = board
        self.current_position = current_position

        self.board_row_size = row_size
        self.board_col_size = col_size

        self.path_to_parent = []

        self.moves = ['L', 'R', 'U', 'D']
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

        return -1, -1

    def available_moves(self, current_state: (int, int)) -> list:
        valid_moves = []
        for _move in self.moves:
            if self.move_validity(current_state, _move) != (-1, -1):
                valid_moves.append(_move)

        return valid_moves

    def update_target(self, position: (int, int)) -> None:
        row_index, col_index = position
        self.board[row_index][col_index] = self.board[row_index][col_index][0]

    def update_energy(self, position: (int, int)) -> None:
        row_index, col_index = position

        extra_energy_cell: str = self.board[row_index][col_index]

        if len(extra_energy_cell) > 1:

            if extra_energy_cell[1] in self.extra_energy.keys():

                # remove extra energy from board
                self.board[row_index][col_index] = extra_energy_cell[0]

                # update energy
                self.energy += self.extra_energy.get(extra_energy_cell[1])
                self.energy -= int(extra_energy_cell[0])
                return

            elif extra_energy_cell[1] in self.other_notations:
                self.energy -= int(extra_energy_cell[0])
                return

        cell_energy_consume = int(self.board[row_index][col_index])
        self.energy -= cell_energy_consume

    def __str__(self) -> str:
        row, col = self.current_position
        return self.board[row][col]

    def is_goal(self):
        # Modify this method based on your specific goal
        # For example, if the goal is to visit all targets, check if all targets are visited
        return all('T' not in row for row in self.board)


def dfs(start_node: Board) -> Optional[Board]:
    visited = set()
    stack = [start_node]

    while stack:
        current_node = stack.pop()
        print(current_node)

        curr_pos = current_node.current_position
        current_node.update_energy(curr_pos)

        if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
            current_node.update_target(curr_pos)
            return current_node

        moves = current_node.available_moves(curr_pos)
        for move in moves:
            child_node = Board(
                current_node.board,
                current_node.board_row_size,
                current_node.board_col_size,
                current_node.current_position,
                current_node.energy
            )
            for _move in current_node.path_to_parent:
                child_node.add_path(_move)

            child_node.add_path(move)

            new_position = current_node.move_validity(curr_pos, move)
            child_node.current_position = new_position

            if child_node not in visited:
                stack.append(child_node)

        visited.add(current_node)

    return None


class Tree:
    def __init__(self):
        self.tree: dict[Board: list[Board]] = dict()

    def add_edge(self, root: Board, dest: Board):
        if root in self.tree:
            self.tree.get(root).append(dest)

        else:
            self.tree[root] = [dest]

    def bfs(self, root_node: Board) -> Optional[Board]:
        queue = deque([root_node])
        visited = set()

        while queue:
            current_node = queue.popleft()

            curr_pos = current_node.current_position
            current_node.update_energy(curr_pos)

            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                current_node.update_target(curr_pos)
                return current_node

            moves = current_node.available_moves(curr_pos)
            for move in moves:
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    current_node.current_position,
                    current_node.energy
                )
                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)

                child_node.add_path(move)
                new_position = current_node.move_validity(curr_pos, move)
                child_node.current_position = new_position

                if child_node not in visited:
                    queue.append(child_node)
                    visited.add(child_node)

                self.add_edge(current_node, child_node)

            visited.add(current_node)

        return None

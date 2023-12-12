from typing import Self
from collections import deque



# TODO: Posistion class


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
        
        return (-1, -1)
    
    def available_moves(self, current_state: (int, int)) -> list:
        valid_moves = []
        for _move in self.moves:
            if self.move_validity(current_state, _move) != (-1, -1):
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

    def is_remaining_target(self) -> bool:
        is_T = False
        for i in self.board:
            for j in i:
                if 'T' in j:
                    is_T = True

        return is_T


class Tree:
    def __init__(self):
        self.tree: dict[Board: list[Board]] = dict()
    
    def add_edge(self, root: Board, dest: Board):
        if root in self.tree:
            self.tree.get(root).append(dest)

        else:
            self.tree[root] = [dest]

    def dfs(self, start_node: Board) -> Board | None:
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


    def bfs(self, root_node: Board, initial_position: (int, int)) -> Board | None:
        queue = deque([root_node])
        visited = set([root_node])

        found = 0
        number_of_target = 2
        targets = []
        path_energy = {"path": [], "energy": float('-inf')}

        while queue:
            current_node = queue.popleft()

            curr_pos = current_node.current_position  # (row, col)


            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                consumed_energy = current_node.calculate_path_energy(current_node.path_to_parent, initial_position)
                print(500 - consumed_energy)
                if consumed_energy > path_energy.get('energy'):
                    path_energy['energy'] = consumed_energy
                    path_energy['path'] = current_node.path_to_parent

            moves = current_node.available_moves(curr_pos)  # successor function
            for move in moves:
                new_position = current_node.move_validity(curr_pos, move)

                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    current_node.energy
                )

                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)
                
                child_node.add_path(move)                

                if child_node not in visited:
                    queue.append(child_node)
                    visited.add(child_node)

            

        return path_energy





matrix = [
    ['1R', '1' , '1' , '5', '5', '4' , '2C', '1' , '15', '1B'],
    ['1' , '1' , '5' , '3', '5', '5' , '5' , '4' , '5' , 'X'],
    ['5' , '1I', '1' , '6', '2', '2' , '2' , '1' , '1' , '1T'],
    ['X' , 'X' , '1' , '6', '5', '5' , '2' , '1' , '1' , 'X'],
    ['X' , 'X' , '1' , 'X', 'X', '50', '2' , '1C', '1' , 'X'],
    ['1' , '1' , '1' , '2', '2', '2T', '2' , '1' , '1' , '1']
]

initial_energy = 500   
initial_position = (0, 0)

board = Board(matrix, 6, 10, initial_position, initial_energy)



tree = Tree()
result = tree.bfs(board, initial_position)
print(result)
# for res in result:
#     print(res.path_to_parent)
#     print(res.energy)




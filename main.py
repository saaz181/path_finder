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


# Define a class representing the game board state
class Board:
    def __init__(self, board: list[list], row_size: int, col_size: int, current_position: (int, int), energy: int):
        # Initialize the Board object with the provided parameters
        self.board = board
        self.current_position = current_position
        self.board_row_size = row_size
        self.board_col_size = col_size
        self.path_to_parent = []  # List to store the path to the parent node
        self.moves = ['L', 'D', 'R', 'U']  # Possible moves: Left, Down, Right, Up
        self.extra_energy = {'C': 10, 'B': 5, 'I': 12}  # Dictionary for extra energy values
        self.other_notations = {'R', 'T'}  # Set of other notations
        self.energy = energy  # Current energy level

    # Add a move to the path_to_parent list
    def add_path(self, move):
        self.path_to_parent.append(move)

    # Check if a path to a destination is valid
    def path_validity_check(self, dest: (int, int)) -> bool:
        d_row, d_col = dest
        path_is_valid = True

        # Check if the destination is within the board boundaries and not blocked by an obstacle ('X')
        if d_row > self.board_row_size - 1 or d_col > self.board_col_size - 1 or d_row < 0 or d_col < 0:
            path_is_valid = False
        elif self.board[d_row][d_col] == 'X':
            path_is_valid = False

        return path_is_valid

    # Check if a move is valid and return the new position
    def move_validity(self, current_state: (int, int), op_code: str) -> (int, int):
        row, col = current_state
        dest_state = (-1, -1)

        # Update the destination based on the specified move
        if op_code == 'U':
            dest_state = (row - 1, col)
        elif op_code == 'D':
            dest_state = (row + 1, col)
        elif op_code == 'R':
            dest_state = (row, col + 1)
        elif op_code == 'L':
            dest_state = (row, col - 1)

        # Check if the move is valid
        if self.path_validity_check(dest_state):
            return dest_state

        return (-1, -1)

    # Get a list of available moves from the current state, optionally including their costs
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

    # Update the target cell on the board
    def update_target(self, position: (int, int)) -> None:
        row_index, col_index = position
        self.board[row_index][col_index] = self.board[row_index][col_index][0]

    # Calculate the energy change for a given position
    def calculate_energy(self, position: (int, int)) -> None:
        row_index, col_index = position
        extra_energy_cell: str = self.board[row_index][col_index]
        if len(extra_energy_cell) > 1:
            if extra_energy_cell[1] in self.extra_energy.keys():

                # Remove extra energy from the board
                self.board[row_index][col_index] = extra_energy_cell[0]

                # Update energy based on the extra energy type
                plus_energy = self.extra_energy.get(extra_energy_cell[1])
                minus_energy = int(extra_energy_cell[0])
                energy = plus_energy - minus_energy

                return energy

            elif extra_energy_cell[1] in self.other_notations:
                return -int(extra_energy_cell[0])

        # If no extra energy, consume energy from the cell
        cell_energy_consume = int(self.board[row_index][col_index])
        return -cell_energy_consume
    
    # Calculate the total energy consumed along a given path
    def calculate_path_energy(self, moves: list, current_position: (int, int)) -> int:
        total_energy = 0
        for move in moves:
            position = self.move_validity(current_position, move)
            energy = self.calculate_energy(position)
            total_energy += energy
            current_position = position

        return total_energy

    # String representation of the current cell
    def __str__(self) -> str:
        row, col = self.current_position
        return self.board[row][col]

    # Comparison for sorting based on energy levels
    def __lt__(self, other):
        return self.energy < other.energy

    # Check if there are remaining 'T' (target) cells on the board
    def is_remaining_target(self) -> bool:
        is_T = False
        for i in self.board:
            for j in i:
                if 'T' in j:
                    is_T = True

        return is_T

    # Heuristic function to estimate the cost of reaching the goal
    def heuristic(self) -> int:
        distances = []  # List to store distances to extra energy cells
        targets = []    # List to store distances to target cells
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

    # Function to check if all targets are found along a given path
    def all_targets_found(self, path, targets, initial_position):
        x, y = initial_position  # Initialize current position with the initial position
        targets_found = []  # List to store positions where targets are found during the path traversal

        # Iterate through each move in the path
        for move in path:
            # Update the current position based on the move
            if move == 'R':
                y += 1
            elif move == 'L':
                y -= 1
            elif move == 'U':
                x -= 1
            elif move == 'D':
                x += 1

            current_position = (x, y)  # Calculate the current position after the move

            # Check if the current position contains a target
            if current_position in targets:
                targets_found.append(current_position)

        # Check if all targets have been found by comparing sets of found targets and the original targets
        return set(targets) == set(targets_found)
    
    # Depth-First Search (DFS) algorithm for finding targets on the game board
    def dfs(self, root_node: Board) -> Board | None:
        # Dictionary to track visited nodes
        visited = {}
        # OrderedSet to represent the stack of nodes to explore
        stack = OrderedSet()
        # Add the root node to the stack
        stack.add(root_node)

        # Continue DFS until the stack is empty or all targets are found
        while stack and self.found_targets < self.targets:
            # Pop the top node from the stack
            current_node = stack.pop()

            # Mark the current node as visited
            visited[current_node.current_position] = current_node.energy

            # Get the current position (row, col) of the node
            curr_pos = current_node.current_position

            # Check if the current cell contains a target ('T')
            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                # Update the target cell on the board
                current_node.update_target(current_node.current_position)

                # Increment the count of found targets
                self.found_targets += 1

                # If all targets are found, return the current node
                if self.found_targets >= self.targets:
                    return current_node

                # Continue DFS from the current node
                temp = self.dfs(current_node)
                # Update the path and energy of the current node based on the recursive result
                current_node.path_to_parent = temp.path_to_parent
                current_node.energy = temp.energy

            # Get available moves from the current position (successor function)
            moves = current_node.available_moves(curr_pos)

            # Explore each possible move from the current node
            for move in moves:
                # Calculate the new position based on the move
                new_position = current_node.move_validity(curr_pos, move)
                # Create a child node with the updated position and energy
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    current_node.energy + current_node.calculate_energy(new_position)
                )

                # Copy the path to the parent from the current node to the child node
                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)

                # Add the current move to the path of the child node
                child_node.add_path(move)

                # Check if the child node has not been visited
                if child_node.current_position not in visited:
                    # Add the child node to the stack for further exploration
                    stack.add(child_node)
                    # Mark the child node as visited with its energy value
                    visited[child_node.current_position] = child_node.energy

                # If the child node has been visited and has higher energy, re-add to the stack
                elif child_node.current_position in visited:
                    if child_node.energy >= visited[child_node.current_position]:
                        stack.add(child_node)
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

        # Return the final state after DFS
        return current_node

        # Breadth-First Search (BFS) algorithm for finding targets on the game board
    def bfs(self, root_node: Board) -> Board | None:
        # Initialize a queue with the root node
        queue = deque([root_node])
        # Dictionary to track visited nodes along with their energy values
        visited = {}

        # Continue BFS until the queue is empty or all targets are found
        while queue and self.found_targets < self.targets:
            # Dequeue the front node from the queue
            current_node = queue.popleft()
            # Mark the current node as visited with its energy value
            visited[current_node.current_position] = current_node.energy
            # Get the current position (row, col) of the node
            curr_pos = current_node.current_position

            # Get available moves from the current position (successor function)
            moves = current_node.available_moves(curr_pos)

            # Explore each possible move from the current node
            for move in moves:
                # Calculate the new position based on the move
                new_position = current_node.move_validity(curr_pos, move)
                # Create a child node with the updated position and energy
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    current_node.energy + current_node.calculate_energy(new_position)
                )

                # Copy the path to the parent from the current node to the child node
                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)

                # Add the current move to the path of the child node
                child_node.add_path(move)

                # Check if the current cell contains a target ('T')
                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    # Update the target cell on the board
                    current_node.update_target(current_node.current_position)

                    # Increment the count of found targets
                    self.found_targets += 1

                    # If all targets are found, return the current node
                    if self.found_targets >= self.targets:
                        return current_node

                    # Continue BFS from the current node
                    temp = self.bfs(current_node)
                    # Update the path and energy of the current node based on the recursive result
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

                # Check if the child node has not been visited
                if child_node.current_position not in visited:
                    # Enqueue the child node for further exploration
                    queue.append(child_node)
                    # Mark the child node as visited with its energy value
                    visited[child_node.current_position] = child_node.energy

                # If the child node has been visited and has higher energy, re-enqueue
                elif child_node.current_position in visited:
                    if child_node.energy >= visited[child_node.current_position]:
                        queue.append(child_node)
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

        # Return the final state after BFS
        return current_node

    # Iterative Deepening Search (IDS) algorithm for finding targets on the game board
    def ids(self, root_node: Board, max_depth: int) -> Board | None:
        # Start with a depth limit of 1 and gradually increase until reaching the max depth
        depth_limit = 1
        visited = dict()

        while depth_limit < max_depth:
            # Use recursive depth-limited search to explore the tree within the current depth limit
            result = self._recursive_dls(root_node, depth_limit, visited)

            # If a solution is found, return the result
            if result is not None:
                return result

            # Increment the depth limit for the next iteration
            depth_limit += 1

    # Helper function for recursive depth-limited search
    def _recursive_dls(self, current_node: Board, depth_limit: int, visited: dict) -> Board | None:
        # Check if the depth limit has been reached
        if depth_limit == 0:
            return None  # Reached depth limit, no solution found at this level

        # Get the current position (row, col) of the node
        curr_pos = current_node.current_position
        # Get available moves from the current position (successor function)
        moves = current_node.available_moves(curr_pos)

        # Mark the current node as visited with its energy value
        if current_node.current_position not in visited:
            visited[current_node.current_position] = current_node.energy
                    
        # If the current position has been visited, update with higher energy value
        elif current_node.current_position in visited:
            if current_node.energy >= visited[current_node.current_position]:
                visited[current_node.current_position] = current_node.energy
                # Continue the search with the updated energy value
                self._recursive_dls(current_node, depth_limit - 1, visited)

        # Explore each possible move from the current node
        for move in moves:
            # Calculate the new position based on the move
            new_position: (int, int) = current_node.move_validity(curr_pos, move)
            # Calculate the energy of the child node
            child_energy: int = current_node.energy + current_node.calculate_energy(new_position)
            
            # Create a child node with the updated position and energy
            child_node = Board(
                current_node.board,
                current_node.board_row_size,
                current_node.board_col_size,
                new_position,
                child_energy
            )

            # Copy the path to the parent from the current node to the child node
            for _move in current_node.path_to_parent:
                child_node.add_path(_move)

            # Add the current move to the path of the child node
            child_node.add_path(move)

            # Check if the current cell contains a target ('T')
            if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                # Update the target cell on the board
                current_node.update_target(current_node.current_position)

                return current_node

            # Recursively call the function with the child node and updated depth limit
            result = self._recursive_dls(child_node, depth_limit - 1, visited)

            # If a solution is found, return the result
            if result is not None:
                return result

        # No solution found at this level
        return None

    # Uniform Cost Search (UCS) algorithm for finding targets on the game board
    def ucs(self, root_node: Board) -> Board:
        # Priority queue with (cost, Board) tuples, initialized with the root node
        priority_queue = [(-0, root_node)]
        visited = {}

        # Continue the search until the priority queue is not empty and all targets are found
        while priority_queue and self.found_targets < self.targets:
            # Pop the node with the minimum cost from the priority queue
            _, current_node = heapq.heappop(priority_queue)
            visited[current_node.current_position] = current_node.energy
            curr_pos = current_node.current_position

            # Get available moves and their costs from the current position (successor function)
            moves = current_node.available_moves(curr_pos, cost=True)

            # Explore each possible move from the current node
            for move, move_cost in moves:
                # Calculate the total cost of reaching the child node
                total_cost = current_node.energy + move_cost
                # Calculate the new position based on the move
                new_position = current_node.move_validity(curr_pos, move)
                # Create a child node with the updated position and total cost
                child_node = Board(
                    current_node.board,
                    current_node.board_row_size,
                    current_node.board_col_size,
                    new_position,
                    total_cost
                )

                # Copy the path to the parent from the current node to the child node
                for _move in current_node.path_to_parent:
                    child_node.add_path(_move)

                # Add the current move to the path of the child node
                child_node.add_path(move)

                # Check if the current cell contains a target ('T')
                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    # Update the target cell on the board
                    current_node.update_target(current_node.current_position)

                    self.found_targets += 1
                    # If all targets are found, return the current node as the solution
                    if self.found_targets >= self.targets:
                        return current_node

                    # Continue the search with the updated path and energy
                    temp = self.ucs(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

                # If the child node is not visited, add it to the priority queue
                if child_node.current_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                    visited[child_node.current_position] = child_node.energy
                
                # If the child node is visited and has higher energy, update it in the priority queue
                elif child_node.current_position in visited:
                    if child_node.energy >= visited[child_node.current_position]:
                        heapq.heappush(priority_queue, (-total_cost, child_node))
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

        # Return the final node reached by the search
        return current_node

    # A* algorithm for finding targets on the game board
    def astar(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = {}  # Dictionary to keep track of visited nodes

        # Continue until the priority queue is empty or all targets are found
        while priority_queue and self.found_targets < self.targets:
            _, current_node = heapq.heappop(priority_queue)  # Pop the node with the lowest cost
            visited[current_node.current_position] = current_node.energy  # Mark the node as visited
            curr_pos = current_node.current_position  # Current position (row, col)
            moves = current_node.available_moves(curr_pos, cost=True)  # Get available moves with costs

            # Explore each possible move from the current node
            for move, move_cost in moves:
                total_cost = current_node.energy + move_cost + current_node.heuristic()  # Calculate total cost
                new_position = current_node.move_validity(curr_pos, move)  # Calculate the new position
                
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


                # Check if the new position is not visited or has a lower cost
                if child_node.current_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                    visited[child_node.current_position] = child_node.energy
                    
                elif child_node.current_position in visited:
                    if child_node.energy > visited[child_node.current_position]:
                        heapq.heappush(priority_queue, (-total_cost, child_node))
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

                # Check if the current position contains a target and update the game state
                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    self.found_targets += 1
                    if self.found_targets >= self.targets:
                        return current_node

                    # Recursively apply A* to the updated game state
                    temp = self.astar(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

        return current_node

    # Best-First Search algorithm for finding targets on the game board
    def best_first_search(self, root_node: Board) -> Board:
        priority_queue = [(-0, root_node)]  # Priority queue with (cost, Board) tuples
        visited = {}  # Dictionary to keep track of visited nodes

        # Continue until the priority queue is empty or all targets are found
        while priority_queue and self.found_targets < self.targets:
            _, current_node = heapq.heappop(priority_queue)  # Pop the node with the lowest cost
            visited[current_node.current_position] = current_node.energy  # Mark the node as visited
            curr_pos = current_node.current_position  # Current position (row, col)
            moves = current_node.available_moves(curr_pos, cost=True)  # Get available moves with costs

            # Explore each possible move from the current node
            for move, move_cost in moves:
                total_cost = current_node.energy + current_node.heuristic()  # Calculate total cost
                new_position = current_node.move_validity(curr_pos, move)  # Calculate the new position
                
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


                # Check if the new position is not visited or has a lower cost
                if child_node.current_position not in visited:
                    heapq.heappush(priority_queue, (-total_cost, child_node))
                    visited[child_node.current_position] = child_node.energy
                    
                elif child_node.current_position in visited:
                    if child_node.energy > visited[child_node.current_position]:
                        heapq.heappush(priority_queue, (-total_cost, child_node))
                        visited[child_node.current_position] = child_node.energy
                else:
                    continue

                # Check if the current position contains a target and update the game state
                if 'T' in current_node.board[curr_pos[0]][curr_pos[1]]:
                    current_node.update_target(current_node.current_position)

                    self.found_targets += 1
                    if self.found_targets >= self.targets:
                        return current_node

                    # Recursively apply Best-First Search to the updated game state
                    temp = self.best_first_search(current_node)
                    current_node.path_to_parent = temp.path_to_parent
                    current_node.energy = temp.energy

        return current_node

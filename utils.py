from main import Board, Tree

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
    


def run_bfs():
    global matrix, initial_energy, initial_position
    
    found_targets = 0
    targets = 2
    final_hit = None
    path = []

    while found_targets != targets:

        board = Board(matrix, 6, 10, initial_position, initial_energy)
        tree = Tree()
        result = tree.bfs(board)

        if result is not None:
            found_targets += 1
            initial_position = result.current_position
            initial_energy = result.energy
            matrix = result.board
            final_hit = result
            for move in result.path_to_parent:
                path.append(move)
    
    print(f"{final_hit.energy} - {''.join(path)}")
    print(final_hit.energy)
    

# run_bfs()

def run_dfs():
    global matrix, initial_energy, initial_position

    board = Board(matrix, 6, 10, initial_position, initial_energy)

    tree = Tree()
    result = tree.dfs(board)

    if result:
        print("DFS path:", result.path_to_parent)
        print("Remaining energy:", result.energy)
    else:
        print("No path found.")
    

# run_dfs()


from collections import deque

def bfs(matrix):
    rows, cols = len(matrix), len(matrix[0])
    start, end = None, None

    # Find the start and end nodes
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j][-1] == 'R':
                start = (i, j, 500, 0)  # (row, col, energy, distance)
            elif matrix[i][j][-1] == 'X':
                end = (i, j)

    if not start or not end:
        return "Start or end not found in the matrix."

    queue = deque([start])
    visited = set([start[:2]])  # Store visited nodes (row, col)

    while queue:
        current_row, current_col, current_energy, distance = queue.popleft()

        if (current_row, current_col) == end:
            return distance

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

        for dr, dc in directions:
            new_row, new_col = current_row + dr, current_col + dc

            if 0 <= new_row < rows and 0 <= new_col < cols and (new_row, new_col) not in visited:
                symbol = matrix[new_row][new_col]
                if not symbol:
                    continue

                energy = int(symbol[:-1])

                if symbol[-1] == 'C':
                    energy += 10
                elif symbol[-1] == 'B':
                    energy += 5
                elif symbol[-1] == 'I':
                    energy += 12

                new_energy = current_energy + energy
                new_distance = distance + 1

                if new_energy > 0:
                    # Remove 'B', 'I', 'C' from the board
                    if symbol[-1] in ['B', 'I', 'C']:
                        matrix[new_row][new_col] = '0'
                    
                    queue.append((new_row, new_col, new_energy, new_distance))
                    visited.add((new_row, new_col))

    return "No valid path found."

matrix = [
    ['1R', '1', '1', '5', '5', '4', '2C', '1', '15', '1B'],
    ['1', '1', '5', '3', '5', '5', '5', '4', '5', 'X'],
    ['5', '1I', '1', '6', '2', '2', '2', '1', '1', '1T'],
    ['X', 'X', '1', '6', '5', '5', '2', '1', '1', 'X'],
    ['X', 'X', '1', 'X', 'X', '50', '2', '1C', '1', 'X'],
    ['1', '1', '1', '2', '2', '2T', '2', '1', '1', '1']
]

result = bfs(matrix)
print(result)
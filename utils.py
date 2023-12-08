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
    # print(final_hit.energy)
    

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
    

run_dfs()
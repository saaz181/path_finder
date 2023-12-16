from main import Board, Tree
import time
import copy




# getting user input
def get_input():
    """
    Function to get the input from the user
    """

    targetN = 0
    x, y = map(int, input().split())
    input_matrix = []
    
    # preparing the matrix
    for i in range(x):
        value = list(input().split())
    
        for i in value:
            if 'T' in i:
                targetN +=1
    
        input_matrix.append(value)
    # our default energy
    
    energy = 500
    return input_matrix, x, y



def run_bfs(matrix: list[list], row_size: int, col_size: int, position: (int, int), energy: int) -> None:
    print("\n############ BFS ############")
    
    s_row, s_col = position
    start_point = int(matrix[s_row][s_col].replace('R', ''))
    energy -= start_point

    board = Board(matrix, row_size, col_size, position, energy)
    tree = Tree(matrix)

    start = time.time()
    bfs_result = tree.bfs(board)
    end = time.time()

    if tree.all_targets_found(bfs_result.path_to_parent, tree.target_numerator(matrix), position):
        print(f"{bfs_result.energy}, {''.join(bfs_result.path_to_parent)}")
    else:
        print("There is no route!!!")
        
    print(f"Time of BFS execution: {end - start}(s)")

    print("###########################\n")
    
    

def run_dfs(matrix: list[list], row_size: int, col_size: int, position: (int, int), energy: int):
    print("\n############ DFS ############")

    s_row, s_col = position
    start_point = int(matrix[s_row][s_col].replace('R', ''))
    energy -= start_point
    
    board = Board(matrix, row_size, col_size, position, energy)
    tree = Tree(matrix)

    start = time.time()

    try:
        dfs_result = tree.dfs(board)
        print(f"{dfs_result.energy}, {''.join(dfs_result.path_to_parent)}")

    except IndexError:
        print("There is no route!!!")

    end = time.time()

    print(f"time of DFS execution: {end - start}(s)")
    
    print("###########################\n")
    

def run_ids(matrix: list[list], row_size: int, col_size: int, position: (int, int), energy: int, max_depth: int):
    print("############ IDS ############")

    s_row, s_col = position
    start_point = int(matrix[s_row][s_col].replace('R', ''))
    energy -= start_point
    
    board = Board(matrix, row_size, col_size, position, energy)
    tree = Tree(matrix)

    targets = tree.targets
    found_target = 0

    start = time.time()
    initial_board = board
    try:
        while found_target != targets:
            ids_result = tree.ids(initial_board, max_depth)
            initial_board = ids_result
            found_target += 1

        print(f"{initial_board.energy}, {''.join(initial_board.path_to_parent)}")
    
    except AttributeError:
        print('There is no route!!!')

    end = time.time()

    print(f"time of IDS execution: {end - start}(s)")
    
    print("###########################")

def run_ucs(matrix: list[list], row_size: int, col_size: int, position: (int, int), energy: int):
    print("\n############ UCS ############")
    
    s_row, s_col = position
    start_point = int(matrix[s_row][s_col].replace('R', ''))
    energy -= start_point

    board = Board(matrix, row_size, col_size, position, energy)
    tree = Tree(matrix)

    start = time.time()
    ucs_result = tree.ucs(board)
    end = time.time()

    if tree.all_targets_found(ucs_result.path_to_parent, tree.target_numerator(matrix), position):
        print(f"{ucs_result.energy}, {''.join(ucs_result.path_to_parent)}")
    else:
        print("There is no route!!!")

    print(f"time of UCS execution: {end - start}(s)")
    
    print("###########################\n")

def run_A_star(matrix: list[list], row_size: int, col_size: int, position: (int, int), energy: int):
    print("\n############ A* ############")

    s_row, s_col = position
    start_point = int(matrix[s_row][s_col].replace('R', ''))
    energy -= start_point

    matrix_2 = copy.deepcopy(matrix)
    
    board = Board(matrix, row_size, col_size, position, energy)
    board1 = Board(matrix_2, row_size, col_size, position, energy)
    
    tree = Tree(matrix)

    start = time.time()
    heuristic_dic = board.heuristic(matrix)
    astar_result = tree.astar(board, heuristic_dic)
    end = time.time()

    if tree.all_targets_found(astar_result.path_to_parent, tree.target_numerator(matrix), position):
        energy_astar = board1.calculate_path_energy(astar_result.path_to_parent, position)
        print(f"{energy + energy_astar}, {''.join(astar_result.path_to_parent)}")
    else:
        print("There is no route!!!")

    print(f"time of A* execution: {end - start}(s)")
    
    print("###########################\n")


def run_best_first_search(matrix: list[list], row_size: int, col_size: int, position: (int, int), energy: int):
    print("\n############ Best First Search ############")

    s_row, s_col = position
    start_point = int(matrix[s_row][s_col].replace('R', ''))
    energy -= start_point
    
    matrix_3 = copy.deepcopy(matrix)

    board = Board(matrix, row_size, col_size, position, energy)
    board2 = Board(matrix_3, row_size, col_size, position, energy)
    tree = Tree(matrix)

    start = time.time()
    heuristic_dic = board.heuristic(matrix)
    best_first_search_result = tree.best_first_search(board, heuristic_dic)
    end = time.time()

    if tree.all_targets_found(best_first_search_result.path_to_parent, tree.target_numerator(matrix), position):
        energy_best_first_search = board2.calculate_path_energy(best_first_search_result.path_to_parent, position)
        print(f"{energy + energy_best_first_search}, {''.join(best_first_search_result.path_to_parent)}")
    
    else:
        print("There is no route!!!")

    print(f"time of Best First Search execution: {end - start}(s)")
    
    print("###########################\n")


if __name__ == '__main__':

    default_input = input('Do you want to use default matrix? (Y/N): ')

    if default_input == 'N':
        d_matrix, row_size, col_size = get_input()

        matrix_bfs = copy.deepcopy(d_matrix)
        matrix_dfs = copy.deepcopy(d_matrix)
        matrix_ids = copy.deepcopy(d_matrix)
        max_depth_limit = 12

        matrix_ucs = copy.deepcopy(d_matrix)
        matrix_astar = copy.deepcopy(d_matrix)
        matrix_best_first_search = copy.deepcopy(d_matrix)

        board_row_size_bfs = board_row_size_dfs = board_row_size_ids = board_row_size_ucs = board_row_size_astar = board_row_size_best_first_search = row_size - 1
        board_col_size_bfs = board_col_size_dfs = board_col_size_ids = board_col_size_ucs = board_col_size_astar = board_col_size_best_first_search = col_size - 1
    
    
    else:
        matrix_bfs = [
        ['1R', '1' , '1' , '5', '5', '4' , '2C', '1' , '15', '1B'],
        ['1' , '1' , '5' , '3', '5', '5' , '5' , '4' , '5' , 'X'],
        ['5' , '1I', '1' , '6', '2', '2' , '2' , '1' , '1' , '1T'],
        ['X' , 'X' , '1' , '6', '5', '5' , '2' , '1' , '1' , 'X'],
        ['X' , 'X' , '1' , 'X', 'X', '50', '2' , '1C', '1' , 'X'],
        ['1' , '1' , '1' , '2', '2', '2T', '2' , '1' , '1' , '1']
    ]

        board_row_size_bfs = len(matrix_bfs)
        board_col_size_bfs = len(matrix_bfs[0]) 

        
        matrix_dfs = [
        ['1R', '1' , '1' , '5', '5', '4' , '2C', '1' , '15', '1B'],
        ['1' , '1' , '5' , '3', '5', '5' , '5' , '4' , '5' , 'X'],
        ['5' , '1I', '1' , '6', '2', '2' , '2' , '1' , '1' , '1T'],
        ['X' , 'X' , '1' , '6', '5', '5' , '2' , '1' , '1' , 'X'],
        ['X' , 'X' , '1' , 'X', 'X', '50', '2' , '1C', '1' , 'X'],
        ['1' , '1' , '1' , '2', '2', '2T', '2' , '1' , '1' , '1']
    ]

        board_row_size_dfs = len(matrix_dfs)
        board_col_size_dfs = len(matrix_dfs[0]) 
    
        matrix_ids = [
        ['1R', '1' , '1' , '5', '5', '4' , '2C', '1' , '15', '1B'],
        ['1' , '1' , '5' , '3', '5', '5' , '5' , '4' , '5' , 'X'],
        ['5' , '1I', '1' , '6', '2', '2' , '2' , '1' , '1' , '1T'],
        ['X' , 'X' , '1' , '6', '5', '5' , '2' , '1' , '1' , 'X'],
        ['X' , 'X' , '1' , 'X', 'X', '50', '2' , '1C', '1' , 'X'],
        ['1' , '1' , '1' , '2', '2', '2T', '2' , '1' , '1' , '1']
    ]

        board_row_size_ids = len(matrix_ids)
        board_col_size_ids = len(matrix_ids[0])
        max_depth_limit = 12

        matrix_ucs = [
        ['1R', '1' , '1' , '5', '5', '4' , '2C', '1' , '15', '1B'],
        ['1' , '1' , '5' , '3', '5', '5' , '5' , '4' , '5' , 'X'],
        ['5' , '1I', '1' , '6', '2', '2' , '2' , '1' , '1' , '1T'],
        ['X' , 'X' , '1' , '6', '5', '5' , '2' , '1' , '1' , 'X'],
        ['X' , 'X' , '1' , 'X', 'X', '50', '2' , '1C', '1' , 'X'],
        ['1' , '1' , '1' , '2', '2', '2T', '2' , '1' , '1' , '1']
    ]

        board_row_size_ucs = len(matrix_ucs)
        board_col_size_ucs = len(matrix_ucs[0]) 

        matrix_astar = [
        ['1R', '1' , '1' , '5', '5', '4' , '2C', '1' , '15', '1B'],
        ['1' , '1' , '5' , '3', '5', '5' , '5' , '4' , '5' , 'X'],
        ['5' , '1I', '1' , '6', '2', '2' , '2' , '1' , '1' , '1T'],
        ['X' , 'X' , '1' , '6', '5', '5' , '2' , '1' , '1' , 'X'],
        ['X' , 'X' , '1' , 'X', 'X', '50', '2' , '1C', '1' , 'X'],
        ['1' , '1' , '1' , '2', '2', '2T', '2' , '1' , '1' , '1']
    ]

        board_row_size_astar = len(matrix_astar)
        board_col_size_astar = len(matrix_astar[0]) 

        matrix_best_first_search = [
        ['1R', '1' , '1' , '5', '5', '4' , '2C', '1' , '15', '1B'],
        ['1' , '1' , '5' , '3', '5', '5' , '5' , '4' , '5' , 'X'],
        ['5' , '1I', '1' , '6', '2', '2' , '2' , '1' , '1' , '1T'],
        ['X' , 'X' , '1' , '6', '5', '5' , '2' , '1' , '1' , 'X'],
        ['X' , 'X' , '1' , 'X', 'X', '50', '2' , '1C', '1' , 'X'],
        ['1' , '1' , '1' , '2', '2', '2T', '2' , '1' , '1' , '1']
    ]

        board_row_size_best_first_search = len(matrix_best_first_search)
        board_col_size_best_first_search = len(matrix_best_first_search[0]) 

    
    initial_energy = 500   
    initial_position = (0, 0)
    
    run_bfs(matrix_bfs, board_row_size_bfs, board_col_size_bfs, initial_position, initial_energy)
    run_dfs(matrix_dfs, board_row_size_dfs, board_col_size_dfs, initial_position, initial_energy)
    run_ids(matrix_ids, board_row_size_ids, board_col_size_ids, initial_position, initial_energy, max_depth_limit)
    run_ucs(matrix_ucs, board_row_size_ucs, board_col_size_ucs, initial_position, initial_energy)
    run_A_star(matrix_astar, board_row_size_astar, board_col_size_astar, initial_position, initial_energy)
    run_best_first_search(matrix_best_first_search, 
                          board_row_size_best_first_search, 
                          board_col_size_best_first_search, 
                          initial_position, 
                          initial_energy)
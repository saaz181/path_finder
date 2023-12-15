from main import Board, Tree
import time

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
    matrix = [
        ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
        ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
        ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
        ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
        ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
        ['1I', '1', '1T', '2', '2', '2T', '2', '1', '1', '1']
    ]
    print("BFS = ")
    board = Board(matrix, 6, 10, initial_position, initial_energy)
    tree = Tree(matrix)
    start = time.time()
    result = tree.bfs(board)
    print(result.path_to_parent)
    print(result.energy)
    print("time elapsed: {:.2f}s".format(time.time() - start))

def run_dfs():
    pass

def run_ids():
    pass

def run_ucs():
    pass

def run_A_star():
    pass
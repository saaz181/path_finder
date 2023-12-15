# Pathfinding Algorithms

This Python script demonstrates the usage of various pathfinding algorithms on a given matrix. The algorithms implemented include Breadth-First Search (BFS), Depth-First Search (DFS), Iterative Deepening Search (IDS), Uniform Cost Search (UCS), A*, and Best-First Search.

## Requirements
- Python 3.x

## Usage

1. Import the necessary classes from `main.py`:

    ```python
    from main import Board, Tree
    import time
    ```

2. Define the matrix and initial parameters for each algorithm:

    ```python
    matrix_bfs = [...]  # Define the matrix for BFS
    matrix_dfs = [...]  # Define the matrix for DFS
    matrix_ids = [...]  # Define the matrix for IDS
    matrix_ucs = [...]  # Define the matrix for UCS
    matrix_astar = [...]  # Define the matrix for A*
    matrix_best_first_search = [...]  # Define the matrix for Best-First Search

    board_row_size = ...  # Define the row size of the matrix
    board_col_size = ...  # Define the column size of the matrix
    initial_position = ...  # Define the initial position (row, col)
    initial_energy = ...  # Define the initial energy
    max_depth_ids = ...  # Define the maximum depth for IDS
    ```

3. Run each algorithm:

    ```python
    run_bfs(matrix_bfs, board_row_size, board_col_size, initial_position, initial_energy)
    run_dfs(matrix_dfs, board_row_size, board_col_size, initial_position, initial_energy)
    run_ids(matrix_ids, board_row_size, board_col_size, initial_position, initial_energy, max_depth_ids)
    run_ucs(matrix_ucs, board_row_size, board_col_size, initial_position, initial_energy)
    run_A_star(matrix_astar, board_row_size, board_col_size, initial_position, initial_energy)
    run_best_first_search(matrix_best_first_search, 
                          board_row_size, 
                          board_col_size, 
                          initial_position, 
                          initial_energy)
    ```

## Results

The script will print the energy consumption and the optimal path found by each algorithm along with the execution time.

Note: Make sure to customize the matrix, initial parameters, and other configurations based on your specific problem.

# matrix = [
#     ['1R', '1', '1', '5', '5', '4', '2C', '1', '15', '1B'],
#     ['1', '1', '5', '3', '5', '5', '5', '4', '5', 'X'],
#     ['5', '1I', '1', '6', '2', '2', '2', '1', '1', '1T'],
#     ['X', 'X', '1', '6', '5', '5', '2', '1', '1', 'X'],
#     ['X', 'X', '1', 'X', 'X', '50', '2', '1C', '1', 'X'],
#     ['1', '1', '1', '2', '2', '2T', '2', '1', '1', '1']
# ]
initial_energy = 500
initial_position = (0, 0)

####### BFS #######
# matrix = [
#     ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
#     ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
#     ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
#     ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
#     ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
#     ['1I', '1', '1T', '2', '2', '2T', '2', '1', '1', '1']
# ]
# print("BFS = ")
# board = Board(matrix, 6, 10, initial_position, initial_energy)
# tree = Tree(matrix)
# start = time.time()
# result = tree.bfs(board)
# print(result.path_to_parent)
# print(result.energy)
# print("time elapsed: {:.2f}s".format(time.time() - start))
# ####### DFS #######
# matrix = [
#     ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
#     ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
#     ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
#     ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
#     ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
#     ['1I', '1', '1T', '2', '2', '2T', '2', '1', '1', '1']
# ]

# print("DFS = ")
# board2 = Board(matrix, 6, 10, initial_position, initial_energy)
# tree2 = Tree(matrix)
# start = time.time()
# result2 = tree2.dfs(board2)
# print(result2.path_to_parent)
# print(result2.energy)
# print("time elapsed: {:.2f}s".format(time.time() - start))


# ####### IDS ########
matrix = [
    ['1R', '1' , '1' , '5', '5', '4' , '2C', '1' , '15', '1B'],
    ['1' , '1' , '5' , '3', '5', '5' , '5' , '4' , '5' , 'X'],
    ['5' , '1I', '1' , '6', '2', '2' , '2' , '1' , '1' , '1T'],
    ['X' , 'X' , '1' , '6', '5', '5' , '2' , '1' , '1' , 'X'],
    ['X' , 'X' , '1' , 'X', 'X', '50', '2' , '1C', '1' , 'X'],
    ['1' , '1' , '1' , '2', '2', '2T', '2' , '1' , '1' , '1']
]

# print("IDS = ")
# board_ids = Board(matrix, 6, 10, initial_position, initial_energy)

# tree_ids = Tree(matrix)
# targets = tree_ids.targets
# max_depth_limit = 50
# # print(tree_ids.ids(board_ids, max_depth_limit).path_to_parent)
# found = 0
# initial_node = board_ids

# path = []
# while found != targets:
#     result = tree_ids.ids(initial_node, max_depth_limit)
#     path = result.path_to_parent
#     # for _move in result.path_to_parent:
#     #     path.append(_move)
#     initial_node = result
#     found += 1

# print(path)
# print(initial_node.energy)


    

# result_ids = tree_ids.ids(board_ids, max_depth_limit)
# print(result_ids.path_to_parent)
# print(result_ids.energy)




# ####### UCS #######
# matrix = [
#     ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
#     ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
#     ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
#     ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
#     ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
#     ['1I', '1', '1T', '2', '2', '2T', '2', '1', '1', '1']
# ]
# print("UCS = ")
# board3 = Board(matrix, 6, 10, initial_position, initial_energy)
# tree3 = Tree(matrix)
# start = time.time()
# result3 = tree3.ucs(board3)
# print(result3.path_to_parent)
# print(result3.energy)
# print("time elapsed: {:.2f}s".format(time.time() - start))

# ####### A* #######
# matrix = [
#     ['1R', 'X' , '1' , '5', '5', '4' , '2C', '1' , '15', '1B'],
#     ['1' , 'X' , '5' , '3', '5', '5' , '5' , '4' , '5' , 'X'],
#     ['5' , '1I', '1' , '6', '2', '2' , '2' , '1' , '1' , '1T'],
#     ['X' , 'X' , '1' , '6', '5', '5' , '2' , '1' , '1' , 'X'],
#     ['X' , 'X' , '1' , 'X', 'X', '50', '2' , '1C', '1' , 'X'],
#     ['1' , '1' , '1' , '2', '2', '2T', '2' , '1' , '1' , '1']
# ]

# print("َA* = ")
# board4 = Board(matrix, 6, 10, initial_position, initial_energy)
# tree4 = Tree(matrix)
# start = time.time()
# result = tree4.astar(board4)
# print(result.path_to_parent)
# print(result.energy)
# print("time elapsed: {}s".format(time.time() - start))


# ###### Best First Search #####
# ####### A* #######
# matrix = [
#     ['1R', '1', 'X', '5T', '5', '4', '2C', '1', '15', '1B'],
#     ['1', '1', '5', '30', '5', '5', '5', 'X', 'X', 'X'],
#     ['X', 'X', '1', 'X', 'X', '2', '2', 'X', '1', '1T'],
#     ['2I', '5', '1', '6', '5', '5', '2', '1', '1', 'X'],
#     ['X', 'X', 'X', 'X', 'X', '50', '2', '1C', 'X', 'X'],
#     ['1I', '1', '1T', '2', '2', '2T', '2', '1', '1', '1']
# ]

# print("Best First Search* = ")
# board_best_first_search = Board(matrix, 6, 10, initial_position, initial_energy)
# tree_best_first_search = Tree(matrix)
# result_best_first_search = tree_best_first_search.best_first_search(board_best_first_search)
# print(result_best_first_search.path_to_parent)
# print(result_best_first_search.energy)

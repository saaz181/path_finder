#regex for spliting word from digit
import re

def extract_digit_and_word(string):
    """
    our matrix value might be combination of letters and digits so we might need this to parse it
    """
    match = re.match(r"(\d+)(\w+)", string)

    if match:
        digit_part = match.group(1)
        word_part = match.group(2)
        return digit_part, word_part
    else:
        return None, None

#getting user input
def get_input():
    """
    Function to get the input from the user
    """

    targetN =0
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
    return input_matrix, energy ,targetN


def find_starting_point(grid, energy):
    """
    find our starting point to traverse
    """
    for row_idx, row in enumerate(grid):
        for col_idx, cell_value in enumerate(row):
            digit, word = extract_digit_and_word(cell_value)
            if word == 'R':
                energy -= int(digit)
                position = (row_idx, col_idx)
                return Node(None, 0, position, energy,'R')

    return None  # Return None if starting point not found

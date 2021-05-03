#!/usr/local/bin/python3
#
# arrange_pichus.py : arrange agents on a grid, avoiding conflicts
#
# Submitted by : [TUSHAR KANT SAMANTARAY, TSAMANT]
#
# Based on skeleton code in CSCI B551, Fall 2020
#


import sys

# Parse the map from a given filename
def parse_map(filename):
	with open(filename, "r") as f:
		return [[char for char in line] for line in f.read().rstrip("\n").split("\n")]

# Count total # of pichus on board
def count_pichus(board):
    return sum([ row.count('p') for row in board ] )

# Return a string with the board rendered in a human-pichuly format
def printable_board(board):
    return "\n".join([ "".join(row) for row in board])

# Add a pichu to the board at the given position, and return a new board (doesn't change original)
def add_pichu(board, row, col):
    return board[0:row] + [board[row][0:col] + ['p',] + board[row][col+1:]] + board[row+1:]
    

# Get list of successors of given board state
def successors(board):
    return [ add_pichu(board, r, c) for r in range(0, len(board)) for c in range(0,len(board[0])) if board[r][c] == '.' and is_valid(board, r, c)]
    

# check if board is a goal state
def is_goal(board):
    return count_pichus(board) == K 

#Check if the position found by successors function is valid or not for placing the pichu.
def is_valid(board, row, col):
    row_valid = True #Boolean value to check if the current row is valid or not.
    col_valid = True #Boolean value to check if the current column is valid or not.
    #Check right side of the selected row.
    for i in range(row, len(board)):
        if board[i][col] == 'p':
            row_valid = False #Set the row_valid value to false if a pichu is already positioned in that row.
        if board[i][col] == 'X' or board[i][col] == '@':
            break #Stop searching along the row if a wall or @ is found.
    #Check left side of the selected row.
    for i in range(row-1, -1, -1):
        if board[i][col] == 'p':
            row_valid = False
        if board[i][col] == 'X' or board[i][col] == '@':
            break
    #Check down side of the selected column.
    for j in range(col, len(board[0])):
        if board[row][j] == 'p':
            col_valid = False
        if board[row][j] == 'X' or board[i][col] == '@':
            break
    #Check up side of the selected column.
    for j in range(col-1, -1, -1):
        if board[row][j] == 'p':
            col_valid = False
        if board[row][j] == 'X' or board[i][col] == '@':
            break
    
    #Check the diagonals if K = 0.
    if K == 0:
        valid_diagonals = True #Boolean value to check if the diagonals are valid or not.
        # Check upper diagonal on left side 
        for i, j in zip(range(row, -1, -1),  
                        range(col, -1, -1)): 
            if board[i][j] == 'p':
                valid_diagonals = False
            if board[i][j] == 'X' or board[i][j] == '@':
                break
        # Check upper diagonal on right side 
        for i, j in zip(range(row, -1, -1),  
                        range(col, len(board[0]), 1)): 
            if board[i][j] == 'p':
                valid_diagonals = False
            if board[i][j] == 'X' or board[i][j] == '@':
                break
        # Check lower diagonal on left side 
        for i, j in zip(range(row, len(board), 1),  
                        range(col, -1, -1)): 
            if board[i][j] == 'p':
                valid_diagonals = False
            if board[i][j] == 'X' or board[i][j] == '@':
                break
        # Check lower diagonal on right side 
        for i, j in zip(range(row, len(board), 1),  
                        range(col, len(board[0]), 1)): 
            if board[i][j] == 'p':
                valid_diagonals = False
            if board[i][j] == 'X' or board[i][j] == '@':
                break
    
    #Check if both row and column are valid.
    if (row_valid and col_valid):
        if (K == 0): 
            #Check if the diagonals are valid if K = 0.
            if valid_diagonals:
                return True
            else:
                return False
        return True
    else:
        return False
# Solve!
def solve(initial_board):
    fringe = [initial_board]
    visited = [] #Used to track the states that are already visited.
    max_pichus = [] #Save the state with maximum pichus placed. 
    while len(fringe) > 0:
        next_search = fringe.pop()
        if (next_search not in visited):
            visited.append(next_search) #Add the current node being searched to visited list.
            for s in successors( next_search ):
                #If k = 0, check if the current explored successors can place more pichus than the previous stored state.
                if K == 0 and count_pichus(s) > count_pichus(max_pichus): 
                    max_pichus = s
                if is_goal(s):
                    return(s)
                fringe.append(s)
    if K == 0:
        return max_pichus
    else:
        return False
        

# Main Function
if __name__ == "__main__":
    house_map = parse_map('map.txt')
    #house_map=parse_map(sys.argv[1])
    # This is K, the number of agents
    #K = int(sys.argv[2])
    K=10
    print ("Starting from initial board:\n" + printable_board(house_map) + "\n\nLooking for solution...\n")
    solution = solve(house_map)
    print ("Here's what we found:")
    print (printable_board(solution) if solution else "None")

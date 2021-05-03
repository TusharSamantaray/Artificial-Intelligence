#!/usr/local/bin/python3
# solver20.py : 2020 Sliding tile puzzle solver
#
# Code by: Tushar Kant Samantaray - tsamant, Monika Krishnamurthy - monkrish
#
# Based on skeleton code by D. Crandall, September 2020
#
import sys
import numpy as np

MOVES = { "R": (0, -1), "L": (0, 1), "D": (-1, 0), "U": (1,0) }
ROWS = 4
COLS = 5
GOAL = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]

# shift a specified row left (1) or right (-1)
def shift_row(state, row, dir):
    change_row = state[(row*COLS):(row*COLS+COLS)]
    return ( state[:(row*COLS)] + change_row[-dir:] + change_row[:-dir] + state[(row*COLS+COLS):], ("L" if dir == -1 else "R") + str(row+1) )

# shift a specified col up (1) or down (-1)
def shift_col(state, col, dir):
    change_col = state[col::COLS]
    s = list(state)
    s[col::COLS] = change_col[-dir:] + change_col[:-dir]
    return (tuple(s), ("U" if dir == -1 else "D") + str(col+1) )

# Converting state from tuple form to matrix form
# To calculate the manhattan distance
'''def createMatrix(state):
    board = []
    for i in range(ROWS):
        rowList = []
        for j in range(COLS):
            rowList.append(state[ROWS * i + j])
        board.append(rowList)
    return board'''

# Heuristic function to calculate the cost from current state to goal state
# cost = Sum of the Number of misplaced tiles + Sum of the manhattan distance from currevt position to goal position
def manhattan_distance(currstate):
    misplace = 0
    for i in range(len(currstate)):
        if (i + 1) != currstate[i]:
            misplace += 1

    #preState = createMatrix(currstate)
    preState = np.reshape(list(currstate),(ROWS, COLS))
    distance = 0
    for i in range(0, ROWS):
        for j in range(0, COLS):
            for m in range(0, ROWS):
                for n in range(0, COLS):
                    if GOAL[i][j] == preState[m][n]:
                        distance += abs(i-m)%3 + abs(j-n)%2
    return misplace + distance

def printable_board(board):
    return [ ('%3d ')*COLS  % board[j:(j+COLS)] for j in range(0, ROWS*COLS, COLS) ]

# return a list of possible successor states
def successors(state):
    return [ (shift_row(state, row, dir)) for dir in (-1,1) for row in range(0, ROWS) ] + \
           [ (shift_col(state, col, dir)) for dir in (-1,1) for col in range(0, COLS) ]

# check if we've reached the goal
def is_goal(state):
    return sorted(state[:-1]) == list(state[:-1])

# The solver!
def solve(initial_board):
    fringe = [(initial_board,manhattan_distance(initial_board),"")] #initialising the fringe with nitial board, cost and route so far taken
    visited ={} #dictionary to keep track of visited states with cost
    visited[(initial_board)] = 0

    while len(fringe) > 0:
        fringe = sorted(fringe,key = lambda x:x[1]) #sorting the fringe based on cost from lowest to highest
        (state,cost, route_so_far) = fringe.pop(0)
        for (succ, move) in successors(state):
            if is_goal(succ):
                return (route_so_far + " " + move)
            if visited.get(succ) is None:     #if the successor state not visited
                fringe.insert(0,(succ,manhattan_distance(succ) +len(route_so_far)  ,  route_so_far + " " + move))
                visited[succ] = len(route_so_far)
            elif visited.get(succ) > len(route_so_far) : #if the successor state is visited before, update the cost
                visited[succ] = len(route_so_far)
                for (fringeState,cost, moves) in fringe:
                    if fringeState == succ:
                        fringe.remove((fringeState, cost, moves))
                        fringe.insert(0,(succ,manhattan_distance(succ)  + len(route_so_far) , route_so_far + " " + move))
    return False

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: expected a board filename"))

    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state += [ int(i) for i in line.split() ]

    if len(start_state) != ROWS*COLS:
        raise(Exception("Error: couldn't parse start state file"))

    print("Start state: \n" +"\n".join(printable_board(tuple(start_state))))
    print("Solving...")
    route = solve(tuple(start_state))
    routelength = 0
    for l in range(len(route)):
        if(route[l].isalpha()):
            routelength+=1
    print("Solution found in " + str(routelength) + " moves:" + "\n" + route[1:])

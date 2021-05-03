# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 05:06:48 2020
@author: tushar
"""

'''
----------------------NOTES------------------------
1. Capital letters represents pieces of the white player and small letters represents pieces of the black players
2. Generate_valid_boards -> This function take two arguments, the color of the player and the board, and returns all possible configuration of the board based on
   all the possible move the current player color can take for the given board.
3. Generate_valid_moves -> This function generates all posssible valid moves for the current player and takes four arguments, the current position of a piece -- (row, col) 
   on the board, the piece -- P, p, R, r etc. based on player color, color -- the color of the current player for whom moves will be fetched, board -- the current board 
   configuration itself.
4. Generate_new_valid_boards -> This function generates actual configuration of the board from the new generated_valid_moves list. This function takes four arguments, same as 
   Generate_valid_moves.
5. Get_best_move -> This function is used to find the next best move for a given player and a given board configuration.
6. Alphabeta -> Implementation of alpha-beta minimax algorithm to find the next best move.
7. Get_move_details -> Returns the initial and final location of the piece in the best move found.
8. Convert_board_to_string -> Convert the numpy 2-D board to string format as per the expected output.
---------------------------------------------------
'''
import sys
import time
import copy
import numpy as np
import datetime
from piece import Get_Valid_Moves as gm
from piece import Evaluation as ev

_INFINITE = 100000
# Generate all possible board move configurations for a player
def generate_valid_boards(color, board):
    pieces = [] # Select pieces to move based on player
    valid_moves = [] # Track all valid moves
    valid_boards = [] # Track all valid board configurations formed from moving pieces according to generated valid moves
    if (color == 'w'):
        pieces = gm.white_pieces
    else:
        pieces = gm.black_pieces

    # Check every available piece on the board
    for position, piece in np.ndenumerate(board):
        if (piece in pieces): # Consider only the pieces of the current player
            valid_moves = generate_valid_moves(position, piece, color, board) # Generate all valid moves for a selected peice
            if (len(valid_moves) > 0): # Check if any new valid moves are generated
                for new_board in generate_new_valid_board(position, valid_moves, color, board): # Generate new board configurations based on generated valid moves
                    if (not any([(new_board == found_board).all() for found_board in valid_boards])): # Check if a new_board already exists in valid_boards
                        valid_boards.append(new_board)
    return valid_boards

# Generate new valid board configurations based on each generated valid moves
def generate_new_valid_board(position, valid_moves, color, board):
    new_boards = []
    for move in valid_moves:
        temp_board = copy.deepcopy(board) # Work on a copy of the board so we do not lose the original board given as parameter to the function.
        current_piece = temp_board[position[0], position[1]]
        if ((color == 'w' and current_piece in gm.white_pieces) or (color == 'b' and current_piece in gm.black_pieces)): # Make sure opponent bots are not moved
            if ((color == 'w' and current_piece == 'P' and move[0] == 7) or (color == 'b' and current_piece == 'p' and move[0] == 0)):
                current_piece = 'Q' if color == 'w' else 'q' # Convert a parakeet into a quertzal if it reaches the other end of the board
            temp_board[move[0], move[1]] = current_piece
            temp_board[position[0], position[1]] = '.'
        # Add board evaluation code here find the cost of this move -- Possible places
        if (not np.array_equal(board, temp_board)): # Do not add repeated board configurations
            new_boards.append(temp_board)
    return new_boards

# Generate valid moves
def generate_valid_moves(position, piece, color, board):
    moves = []
    if (piece == 'P' or piece == 'p'):
        moves = gm.get_parakeet_moves(position, color, board)
    elif (piece == 'R' or piece == 'r'):
        moves = gm.get_robin_moves(position, color, board)
    elif (piece == 'B' or piece == 'b'):
        moves = gm.get_bluejay_moves(position, color, board)
    elif (piece == 'N' or piece == 'n'):
        moves = gm.get_nighthawk_moves(position, color, board)
    elif (piece == 'K' or piece == 'k'):
        moves = gm.get_kingfisher_moves(position, color, board)
    elif (piece == 'Q' or piece == 'q'):
        moves = gm.get_quetzal_moves(position, color, board)
    else:
        raise(Exception("Error: Invalid piece"))
    return moves

# Returns the best move for the player for a given board configuration
def get_best_move(board, color, end_limit):
    best_move = None
    best_score = _INFINITE # Initialise the best_score with a very high value
    is_mid_game = ev.is_mid_game(board, color) # Check if the game is in midgame or endgame based on with the kingfisher behaves differently
    for board in generate_valid_boards(color, board):
        # For each generated valid board for the player, check the opponents move
        if (color == 'b'):
            color = 'w'
        if (color == 'w'):
            color = 'b'
        score = alphabeta(board, 3, -_INFINITE, _INFINITE, color, True, end_limit, is_mid_game) # Find the best move for each possible step
        if (score < best_score): # The next best move has a score less than the previous best move
            best_score = score
            best_move = board
            print(convert_board_to_string(best_move)) #Print intermediate best move found yet
    # Checkmate
    if best_move is None:
        return None

    return best_move

# Using alpha-beta minimax algorithm to find the best move till depth 3
def alphabeta(board, depth, a, b, color, maximizing, end_limit, is_mid_game):
    if (depth == 0 or datetime.datetime.now() > end_limit): # Return if depth is 0 or if time is up
        return ev.evaluate(board, color, is_mid_game) # Call the board evaluation function

    if (maximizing): # Check if we are finding the maximum score
        best_score = -_INFINITE
        for board in generate_valid_boards(color, board): # Check all the possible moves
            best_score = max(best_score, alphabeta(board, depth - 1, a, b, color, False, end_limit, is_mid_game)) # Check for the next depth level. If the curret move maximizes, the next one shall minimize
            a = max(a, best_score) # Prune the lower score branch
            if (b <= a): # Return if both score are same
                break
        return best_score
    else:
        best_score = _INFINITE # Take a high value and search for the minimum value
        # Check for opponent's move 
        if (color == 'b'):
            color = 'w'
        if (color == 'w'):
            color = 'b'
        for board in generate_valid_boards(color, board):
            best_score = min(best_score, alphabeta(board, depth - 1, a, b, color, True, end_limit, is_mid_game)) # Check for the next depth level. If the curret move minimizes, the next one shall maximize
            b = min(b, best_score) # Prune the higher score branch
            if (b <= a): # Return if both score are same
                break
        return best_score

# Returns the initial and final position of the moved piece on the board
def get_move_details(board, move):
    initial_position = final_position = 0
    for position, piece in np.ndenumerate(board):
        row = position[0]
        col = position[1]
        if (board[row][col] != move[row][col]):
            if (board[row][col] != '.' and move[row][col] == '.'):
                initial_position = (row, col)
            else:
                final_position = (row, col)
    return initial_position, final_position

# Retuns the board in the form of a string
def convert_board_to_string(board):
    return ''.join(col for row in board for col in row)

if __name__ == "__main__":
    #if(len(sys.argv) != 4):
    #    raise(Exception("Error: Please provide valid inputs"))

    #color = sys.argv[1]
    #board = np.reshape(list(sys.argv[2]),(8, 8))
    #max_time_allowed = float(sys.argv[3]) * 0.90 # Only allow the program of running 90% of the allowed time and reserve 10% of the time for the exit process
    # Hardcoded For Testing
    color = 'b'
    #board = np.reshape(list('RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr'), (8, 8))
    board = np.reshape(list('RNBQKBNR.PPPPPPP........P.......................pppppppprnbqkbnr'), (8, 8))
    #board = np.reshape(list('RNBQKBNR.PPPPPPP........P........p..............p.pppppprnbqkbnr'), (8, 8))
    #board = np.reshape(list('RN..KB....PP.P...Q.........N..P.Pq....pp.....p.r..p.p.b..nb.k.n.'), (8, 8))
    board = np.reshape(board,(8, 8))
    max_time_allowed = 7
    max_time_allowed = max_time_allowed * 0.90
    if (color not in ['w', 'b']):
        raise(Exception("Error: Please provide 'b' or 'w' as current player"))
    start = datetime.datetime.now()
    end_limit = start + datetime.timedelta(0, max_time_allowed)
    best_move = get_best_move(board, color, end_limit)
    end = datetime.datetime.now()
    elapse = end - start
    print(f"It took {elapse} to find the best move.")
    if best_move is None:
        print("There is no valid move to take.")
    else:
        initial_position, final_position = get_move_details(board, best_move)
        print('Moving the ' + gm.get_piece_name(board[initial_position[0], initial_position[1]]) + ' at location (' + str(initial_position[0]), str(initial_position[1]) + ') to (' + str(final_position[0]), str(final_position[1]) + ')')
    print("New_board: ")
    print(best_move) #-> To see the board in 2-D format
    #print(convert_board_to_string(best_move))
import numpy as np

# Generate valid move for all pieces
class Get_Valid_Moves:
    white_pieces = ['P', 'R', 'N', 'B', 'Q', 'K']
    black_pieces = ['p', 'r', 'n', 'b', 'q', 'k']

    def get_piece_name(piece):
        if (piece == 'P' or piece == 'p'):
            return 'Parakeet'
        elif (piece == 'R' or piece == 'r'):
            return 'Robin'
        elif (piece == 'N' or piece == 'n'):
            return 'Nighthawk'
        elif (piece == 'B' or piece == 'b'):
            return 'Bluejay'
        elif (piece == 'K' or piece == 'k'):
            return "Kingfisher"
        elif (piece == 'Q' or piece == 'q'):
            return 'Quetzal'
        else:
            return ''

    # Returns all valid moves of a parakeet
    def get_parakeet_moves(position, color, board):
        possible_moves = []
        row = position[0]
        col = position[1]
        if (color == 'w'):        
            if (board[row + 1][col] == '.'):
                possible_moves.append((row + 1, col))
            if (row == 1
                and board[row + 1][col] == '.'
                and board[row + 2][col] == '.'):
                # Parakeet can move 2 places if it's their first move
                    possible_moves.append((row + 2, col))
            if (col - 1 >= 0 and row + 1 <=7 and board[row + 1][col - 1] in Get_Valid_Moves.black_pieces):
                possible_moves.append((row + 1, col - 1))
            if (col + 1 <= 7 and row + 1 <= 7 and board[row + 1][col + 1] in Get_Valid_Moves.black_pieces):
                possible_moves.append((row + 1, col + 1)) 
        if (color == 'b'):
            if (board[row - 1][col] == '.'):
                possible_moves.append((row - 1, col))
            if (row == 6
                and board[row - 1][col] == '.'
                and board[row - 2][col] == '.'):
                    possible_moves.append((row - 2, col))
            if (col - 1 >= 0 and row - 1 >= 0 and board[row - 1][col - 1] in Get_Valid_Moves.white_pieces):
                possible_moves.append((row - 1, col - 1))
            if (col + 1 <= 7 and row - 1 >= 0 and board[row - 1][col + 1] in Get_Valid_Moves.white_pieces):
                possible_moves.append((row - 1, col + 1))
        
        return possible_moves

    # Returns all valid moves of a robin
    def get_robin_moves(position, color, board):
        possible_moves = []
        row = position[0]
        col = position[1]
        # Move upwards
        for i in range(row - 1, -1, -1):
            p = board[i][col]
            if p == '.':
                possible_moves.append((i, col))
            elif (color == 'w' and p in Get_Valid_Moves.black_pieces):
                possible_moves.append((i, col))
                break
            elif (color == 'b' and p in Get_Valid_Moves.white_pieces):
                possible_moves.append((i, col))
                break
            else:
                break
        # Move downwards       
        for i in range(row + 1, 8, 1):
            p = board[i][col]
            if p == '.':
                possible_moves.append((i, col))
            elif (color == 'w' and p in Get_Valid_Moves.black_pieces):
                possible_moves.append((i, col))
                break
            elif (color == 'b' and p in Get_Valid_Moves.white_pieces):
                possible_moves.append((i, col))
                break
            else:
                break
        # Move right
        for j in range(col + 1, 8, 1):
            p = board[row][j]
            if p == '.':
                possible_moves.append((row, j))
            elif (color == 'w' and p in Get_Valid_Moves.black_pieces):
                possible_moves.append((row, j))
                break
            elif (color == 'b' and p in Get_Valid_Moves.white_pieces):
                possible_moves.append((row, j))
                break
            else:
                break
        # Move left
        for j in range(col - 1, -1, -1):
            p = board[row][j]
            if p == '.':
                possible_moves.append((row, j))
            elif (color == 'w' and p in Get_Valid_Moves.black_pieces):
                possible_moves.append((row, j))
                break
            elif (color == 'b' and p in Get_Valid_Moves.white_pieces):
                possible_moves.append((row, j))
                break
            else:
                break
        return possible_moves

    # Returns all valid moves of a bluejay
    def get_bluejay_moves(position, color, board):
        possible_moves = []
        row = position[0]
        col = position[1]

        # Move upper left diagonal
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            p = board[i][j]
            if p == '.':
                possible_moves.append((i, j))
            elif (color == 'w' and p in Get_Valid_Moves.black_pieces):
                possible_moves.append((i, j))
                break
            elif (color == 'b' and p in Get_Valid_Moves.white_pieces):
                possible_moves.append((i, j))
                break
            else:
                break
        # Move upper right diagonal 
        for i, j in zip(range(row-1, -1, -1), range(col+1, 8, 1)):
            p = board[i][j]
            if p == '.':
                possible_moves.append((i, j))
            elif (color == 'w' and p in Get_Valid_Moves.black_pieces):
                possible_moves.append((i, j))
                break
            elif (color == 'b' and p in Get_Valid_Moves.white_pieces):
                possible_moves.append((i, j))
                break
            else:
                break
        # Move lower left diagonal
        for i, j in zip(range(row, 8, 1), range(col, -1, -1)): 
            p = board[i][j]
            if p == '.':
                possible_moves.append((i, j))
            elif (color == 'w' and p in Get_Valid_Moves.black_pieces):
                possible_moves.append((i, j))
                break
            elif (color == 'b' and p in Get_Valid_Moves.white_pieces):
                possible_moves.append((i, j))
                break
            else:
                break
        # Move lower right diagonal
        for i, j in zip(range(row, 8, 1), range(col, 8, 1)): 
            p = board[i][j]
            if p == '.':
                possible_moves.append((i, j))
            elif (color == 'w' and p in Get_Valid_Moves.black_pieces):
                possible_moves.append((i, j))
                break
            elif (color == 'b' and p in Get_Valid_Moves.white_pieces):
                possible_moves.append((i, j))
                break
            else:
                break
        return possible_moves

    # Returns all valid moves of a nighthawk
    def get_nighthawk_moves(position, color, board):
        possible_moves = []
        row = position[0]
        col = position[1]

        if (color == 'w'):
            if (row - 2 >= 0 and col + 1 <= 7 and board[row - 2][col + 1] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row - 2, col + 1))
            if (row - 1 >= 0 and col + 2 <= 7 and board[row - 1][col + 2] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row - 1, col + 2))
            if (row + 1 <= 7 and col + 2 <= 7 and board[row + 1][col + 2] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row + 1, col + 2))
            if (row + 2 <= 7 and col + 1 <= 7 and board[row + 2][col + 1] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row + 2, col + 1))
            if (row + 2 <= 7 and col - 1 >= 0 and board[row + 2][col - 1] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row + 2, col - 1))
            if (row + 1 <= 7 and col - 2 >= 0 and board[row + 1][col - 2] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row + 1, col - 2))
            if (row - 1 >= 0 and col - 2 >= 0 and board[row - 1][col - 2] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row - 1, col - 2))
            if (row - 2 >= 0 and col - 1 >= 0 and board[row - 2][col - 1] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row - 2, col - 1))
        if (color == 'b'):
            if (row - 2 >= 0 and col + 1 <= 7 and board[row - 2][col + 1] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row - 2, col + 1))
            if (row - 1 >= 0 and col + 2 <= 7 and board[row - 1][col + 2] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row - 1, col + 2))
            if (row + 1 <= 7 and col + 2 <= 7 and board[row + 1][col + 2] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row + 1, col + 2))
            if (row + 2 <= 7 and col + 1 <= 7 and board[row + 2][col + 1] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row + 2, col + 1))
            if (row + 2 <= 7 and col - 1 >= 0 and board[row + 2][col - 1] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row + 2, col - 1))
            if (row + 1 <= 7 and col - 2 >= 0 and board[row + 1][col - 2] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row + 1, col - 2))
            if (row - 1 >= 0 and col - 2 >= 0 and board[row - 1][col - 2] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row - 1, col - 2))
            if (row - 2 >= 0 and col - 1 >= 0 and board[row - 2][col - 1] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row - 2, col - 1))
        
        return possible_moves

    # Returns all valid moves of a kingfisher
    def get_kingfisher_moves(position, color, board):
        possible_moves = []
        row = position[0]
        col = position[1]

        if (color == 'w'):
             # Move up
            if (row - 1 >= 0 and board[row - 1][col] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row - 1, col))
            # Move down
            if (row + 1 <= 7 and board[row + 1][col] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row + 1, col))
            # Move left
            if (col - 1 >= 0 and board[row][col - 1] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row, col - 1))
            # Move right
            if (col + 1 <= 7 and board[row][col + 1] not in Get_Valid_Moves.white_pieces):
                possible_moves.append((row, col + 1))

        if (color == 'b'):
             # Move up
            if (row - 1 >= 0 and board[row - 1][col] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row - 1, col))
            # Move down
            if (row + 1 <= 7 and board[row + 1][col] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row + 1, col))
            # Move left
            if (col - 1 >= 0 and board[row][col - 1] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row, col - 1))
            # Move right
            if (col + 1 <= 7 and board[row][col + 1] not in Get_Valid_Moves.black_pieces):
                possible_moves.append((row, col + 1))

        return possible_moves

    # Returns all valid moves of a quetzal
    def get_quetzal_moves(position, color, board):
        # quetzal is a combination of robin and bluejay
        possible_moves = []
        robin_moves = Get_Valid_Moves.get_robin_moves(position, color, board)
        possible_moves.extend(robin_moves)
        bluejay_moves = Get_Valid_Moves.get_bluejay_moves(position, color, board)
        possible_moves.extend(bluejay_moves)
        return possible_moves

# Contains function and values to evalute a current board
class Evaluation:
    # The tables denote the points scored for the position of the chess pieces on the board.

    PARAKEET_TABLE = np.array([
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 5, 10, 10,-20,-20, 10, 10,  5],
        [ 5, -5,-10,  0,  0,-10, -5,  5],
        [ 0,  0,  0, 20, 20,  0,  0,  0],
        [ 5,  5, 10, 25, 25, 10,  5,  5],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ])

    NIGHTHAWK_TABLE = np.array([
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20,   0,   5,   5,   0, -20, -40],
        [-30,   5,  10,  15,  15,  10,   5, -30],
        [-30,   0,  15,  20,  20,  15,   0, -30],
        [-30,   5,  15,  20,  20,  15,   0, -30],
        [-30,   0,  10,  15,  15,  10,   0, -30],
        [-40, -20,   0,   0,   0,   0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ])

    BLUEJAY_TABLE = np.array([
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10,   5,   0,   0,   0,   0,   5, -10],
        [-10,  10,  10,  10,  10,  10,  10, -10],
        [-10,   0,  10,  10,  10,  10,   0, -10],
        [-10,   5,   5,  10,  10,   5,   5, -10],
        [-10,   0,   5,  10,  10,   5,   0, -10],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ])

    ROBIN_TABLE = np.array([
        [ 0,  0,  0,  5,  5,  0,  0,  0],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [ 5, 10, 10, 10, 10, 10, 10,  5],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ])

    QUETZAL_TABLE = np.array([
        [-20, -10, -10, -5, -5, -10, -10, -20],
        [-10,   0,   5,  0,  0,   0,   0, -10],
        [-10,   5,   5,  5,  5,   5,   0, -10],
        [  0,   0,   5,  5,  5,   5,   0,  -5],
        [ -5,   0,   5,  5,  5,   5,   0,  -5],
        [-10,   0,   5,  5,  5,   5,   0, -10],
        [-10,   0,   0,  0,  0,   0,   0, -10],
        [-20, -10, -10, -5, -5, -10, -10, -20]
    ])

    KINGFISHER_MIDGAME_TABLE = np.array([
        [ 20,  30,  10,   0,   0,  10,  30,  20],
        [ 20,  20,   0,   0,   0,   0,  20,  20],
        [-10, -20, -20, -20, -20, -20, -20, -10],
        [-20, -30, -30, -40, -40, -30, -30, -20],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30]
    ])

    KINGFISHER_ENDGAME_TABLE = np.array([
        [-50, -30, -30, -30, -30, -30, -30, -50],
        [-30, -30,   0,   0,   0,   0, -30, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -20, -10,   0,   0, -10, -20, -30],
        [-50, -40, -30, -20, -20, -30, -40, -50]
    ])

    # Get numercial values of each piece
    def get_piece_value(piece):
        if (piece == ''):
            return 0
        value = 0
        if (piece == 'P' or 'p'):
            value = 100
        elif (piece == 'R' or 'r'):
            value = 500
        elif (piece == 'N' or 'n'):
            value = 320
        elif (piece == 'B' or 'b'):
            value = 330
        elif (piece == 'Q' or 'q'):
            value = 900
        elif (piece == 'K' or 'k'):
            value = 20000
        else:
            raise(Exception("Error: Invalid piece"))
        
        return value

    '''
    Evaluation function -
    Gets the total point of each bot of a current player on the board based on the individual value of each piece 
    and their position points on the board. The idea is to prefer a state wit higher point as a higher points means
    the more number of pieces are located at more favourable locations on the board.
    '''
    def evaluate(board, color, is_mid_game):
        piece = Evaluation.get_piece_score(board, color)
        pieces = Get_Valid_Moves.white_pieces if (color == 'w') else Get_Valid_Moves.black_pieces

        parakeet = Evaluation.get_piece_position_score(board, pieces[0], color, Evaluation.PARAKEET_TABLE)
        robin = Evaluation.get_piece_position_score(board, pieces[1], color, Evaluation.ROBIN_TABLE)
        nighthawk = Evaluation.get_piece_position_score(board, pieces[2], color, Evaluation.NIGHTHAWK_TABLE)
        bluejay = Evaluation.get_piece_position_score(board, pieces[3], color, Evaluation.BLUEJAY_TABLE)        
        quetzal = Evaluation.get_piece_position_score(board, pieces[4], color, Evaluation.QUETZAL_TABLE)


        if (is_mid_game):
            kingfisher = Evaluation.get_piece_position_score(board, pieces[4], color, Evaluation.KINGFISHER_MIDGAME_TABLE)
        else:
            kingfisher = Evaluation.get_piece_position_score(board, pieces[4], color, Evaluation.KINGFISHER_ENDGAME_TABLE)

        return piece + parakeet + robin + bluejay + nighthawk + quetzal + kingfisher

    # Evaluate the current position of a piece
    def get_piece_position_score(board, evaluate_piece, color, table):
        white = black = 0
        for position, piece in np.ndenumerate(board):
            if (piece == evaluate_piece.upper() and piece in Get_Valid_Moves.white_pieces):
                white += table[position[0]][position[1]]
            if (piece == evaluate_piece.lower() and piece in Get_Valid_Moves.black_pieces):
                black += table[7 - position[0]][position[1]]

        return (white - black) if color == 'b' else (black - white)
    
    # Evaluates the value of all the pieces of a player, and returns the difference based on which player is currently playing. 
    def get_piece_score(board, color):
        white = black = 0
        for position, piece in np.ndenumerate(board):
            if (piece in Get_Valid_Moves.white_pieces):
                white += Evaluation.get_piece_value(piece)
            if (piece in Get_Valid_Moves.black_pieces):
                black += Evaluation.get_piece_value(piece)

        return (white - black) if color == 'b' else (black - white)

    # Check if the curent board configuration is in mid-game
    def is_mid_game(board, color):
        white_parakeet_count = np.count_nonzero(board == 'P')
        black_parakeet_count = np.count_nonzero(board == 'p')

        remaining_white_pieces = remaining_black_pieces = 0
        for piece in Get_Valid_Moves.white_pieces:
            remaining_white_pieces += np.count_nonzero(board == piece)
        for piece in Get_Valid_Moves.black_pieces:
            remaining_black_pieces += np.count_nonzero(board == piece)

        '''
        Check if the number of parakeets of the opponent is less than 5 and is same or less than the parakeets of self. 
        Also check if there are less than 9 pieces in total of the opponent for the kingfisher to start playing aggressively.
        The current player should have at least the same amount of parakeets and total pieces to start playing aggressively.
        '''
        if (color == 'w'): 
            if (black_parakeet_count < 5 and black_parakeet_count <= white_parakeet_count and remaining_black_pieces < 9 and remaining_black_pieces <= remaining_white_pieces):
                return False
        if (color == 'b'):
            if (white_parakeet_count < 5 and white_parakeet_count <= black_parakeet_count and remaining_white_pieces < 9 and remaining_white_pieces <= remaining_black_pieces):
                return False
        return True 
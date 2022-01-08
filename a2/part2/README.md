## Approach – <br/>
1.	The first step is to read the command line inputs and create a 2d numpy array representation of the given board in string format. We call the ‘get_best_move() function which generated the next best move for the given board and player. <br/>
2.	In get_best_move() function, we iterate over all valid boards generated for the current player on the current board using generate_valid_boards() functions and find the best move using alphabeta() method. The is_mid_game() function checks if the current board is in midgame or endgame based on which the kingfisher will player either defensively or aggressively (Check the is_mid_game() function description below to see how midgame or endgame is decided).  <br/>
3.	The generate_valid_boards() function enumerate over the 2d numpy representation of the given board, and searches for each valid moves for all the pieces available for the current player using generate_valid_moves() function. Once all valid moves are generated, the generate_new_valid_board() function generates new 2d representation of valid board configurations. And the generate_valid_board() function finally returns all possible next board configuration to the get_best_move() function upon which alphabeta() method is called which searches till depth 3 for the best moves. <br/>
4.	Once all the states are checked, or if the time limit runs out, the alphabeta() function returns the best found move, based on the score from the evaluation() function. <br/>
5.	Finally, convert_board_to_string() transforms the numpy 2d board to an acceptable string format and returns the best move as a string. <br/>
<br/>
## Betsy.py explanation - <br/>
The following set of functions are implemented in Betsy.py file. The implementation in [this](https://github.com/Dirk94/ChessAI/blob/master/ai.py) git repo provided us insight to implement the alphabeta pruning minimax algorithm for a Chess AI.
. However, the implementation there only works for white player and uses external libraries to generate valid moves. We had to make sure our code works for each player and generate the legal moves. <br/>
1.	Generate_valid_boards(color, board) -> This function take two arguments, the color of the player and the board, and returns all possible configuration of the board based on all the possible move the current player color can take for the given board. <br/>
2.	Generate_valid_moves(position, piece, color, board) -> This function generates all posssible valid moves for the current player and takes four arguments, the current position of a piece -- (row, col) on the board, the piece -- P, p, R, r etc. based on player color, color -- the color of the current player for whom moves will be fetched, board -- the current board configuration itself. <br/>
3.	Generate_new_valid_board(position, valid_moves, color, board)  -> This function generates actual configuration of the board from the new generated_valid_moves list. This function takes four arguments, same as Generate_valid_moves. <br/>
4.	Get_best_move(board, color, end_limit) -> This function is used to find the next best move, within a stipulated time, for a given player and a given board configuration. <br/>
5.	Alphabeta(board, depth, a, b, color, maximizing, end_limit, is_mid_game) -> Implementation of alpha-beta minimax algorithm to find the next best move. If maximum 97% of the maximum stipulated time is reached, we return out of the function. The rest 3% time is reserved to print the new board and exit out of the program. <br/>
6.	Get_move_details(board, move) -> Returns the initial and final location of the piece in the best move found. <br/>
7.	Convert_board_to_string(board) -> Convert the numpy 2-D board to string format as per the expected output. <br/>
<br/>
## Pieces.py explanation - <br/>
[This](https://github.com/techwithtim/Online-Chess-Game/blob/master/piece.py) repository provided us initial inspiration to code valid moves for the bots, however we found simpler ways to generate them ourselves. <br/>
The following set of functions are implemented in piece.py file –  <br/>
## Get_Valid_Moves class explanaton - <br/>
Class Get_Valid_Moves: Move generation for all the pieces are implemented here. <br/>
1.	get_piece_name(piece) -> Returns the full name of a given piece. <br/>
2.	get_parakeet_moves(position, color, board) -> Generates all possible parakeet move located at a given position, for a provided player color on the given board configuration. <br/>
3.	get_robin_moves(position, color, board) -> Generates all possible robin move located at a given position, for a provided player color on the given board configuration. <br/>
4.	get_ bluejay _moves (position, color, board) -> Generates all possible bluejay move located at a given position, for a provided player color on the given board configuration. <br/>
5.	get_nighthawk_moves(position, color, board) -> Generates all possible nighthawk move located at a given position, for a provided player color on the given board configuration. <br/>
6.	get_kingfisher_moves(position, color, board) -> Generates all possible kingfisher move located at a given position, for a provided player color on the given board configuration. <br/>
7.	get_quetzal_moves(position, color, board) -> Generates all possible quetzal move located at a given position, for a provided player color on the given board configuration.
8.	List white_pieces -> contains all 6 preassigned characters for the pieces for white player. All the characters are in capital letters. <br/>
9.	List black_pieces -> contains all 6 preassigned characters for the pieces for black player. All the characters are in small letters. <br/>
<br/>
<br/>
## Evaluation Class explanation - <br/>
Class Evaluation: Logic for evaluation of the board are implemented here. The evaluation function implementation is inspired from [This article](https://www.chessprogramming.org/Simplified_Evaluation_Function/). We have used the same piece-square tables as defined here. <br/>
1.	PARAKEET_TABLE -> Contains the position values of parakeets in a board. We reserve the order of each of the board for white player. The initial board is defined for black player. <br/>
2.	NIGHTHAWK_TABLE -> Contains the position values of nighthawks in a board. <br/>
3.	BLUEJAY_TABLE -> Contains the position values of bluejays in a board. <br/>
4.	ROBIN_TABLE -> Contains the position values of robins in a board. <br/>
5.	QUETZAL_TABLE -> Contains the position values of quetzals in a board. <br/>
6.	KINGFISHER_MIDGAME_TABLE -> Contains the position values of kingfishers in a board during mid game. This makes the kingfisher play more defensively in the beginning. <br/>
7.	KINGFISHER_ENDGAME_TABLE -> Contains the position values of kingfishers in a board during end game. This makes the kingfisher play more aggressively towards the end. This following article inspired the idea of end and mid game moves for the kingfisher [end game](https://www.chessprogramming.org/Endgame/) strategy<br/>
8.	get_piece_value(piece) -> Get the individual value of a piece on the board. <br/>
9.	evaluate (board, color, is_mid_game) -> Evaluates the total cost of a board used in alpha-beta pruning to decide the best move, for a given board and player color. The is_mid_game parameter lets the function know if it is still mid_game or end_game to decide to choose between KINGFISHER_MIDGAME_TABLE and KINGFISHER_ENDGAME_TABLE. <br/>
10.	get_piece_position_score (board, evaluation_piece, color, table) -> Returns the position value of a piece on a current board configuration, based on the player color. The table parameter lets the function know which table to look for position value from based on the piece being evaluated. <br/>
11.	get_piece_score(board, color) -> Evaluates the value of all the pieces of a player, and returns the difference based on which player is currently playing. <br/>
12.	Is_mid_game(board, color) -> This function checks if it is still mid-game or end-game. During the endgame, the kingfisher player more aggressively, taking position towards the middle half of the board. (Note: Since we could not find any details and conclusive instruction about how to differentiate between mid and end game, we implemented a silly implementation of our own. In case the opponent has less than 5 parakeets remaining on the board and the total remaining pieces of the opponent is less than 9, the kingfisher starts taking aggressive moves. Also, the current player should have the same or a greater number of parakeets and total pieces to make the kingfisher take aggressive moves. <br/>
<br/>
## Additional Resources referred - <br/>
Other helpful resources that provided us inspiration – <br/>
1.	https://www.freecodecamp.org/news/simple-chess-ai-step-by-step-1d55a9266977/ <br/>
2.	https://impythonist.wordpress.com/2017/01/01/modeling-a-chessboard-and-mechanics-of-its-pieces-in-python/ <br/>
3.	https://towardsdatascience.com/implementing-a-chess-engine-from-scratch-be38cbdae91 <br/>
4.	https://www.freecodecamp.org/news/how-i-built-my-one-person-open-source-project/ <br/>
5.	https://www.naftaliharris.com/blog/chess/ <br/>
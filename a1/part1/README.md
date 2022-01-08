Approach -  
 The solution to the problem is to find the shortest sequence of moves to arrange numbers in an order on the 5*4 board. The initial code provided implemented BFS at first. The aim was to find the better heuristic  function to reach the goal from the given board with least moves  possible. Initially we started by calculating misplaced numbers and  added additional cost field to the fringe to keep track of the cost and  sorted the fringe based on cost from lowest to highest. Even though the  misplaced numbers heuristic function improved the performance, it failed to give results within acceptable amount of time for more complicated  boards. So, we tried to implement Manhattans distance for every position to its goal position. We added every distance and saved as a cost to  reach the goal state. We also tried to add states with only cost lesser  than previous state, but it failed to reach the goal state as we were  limiting the generation of few successor states. The general Manhattan  distance was not accurate for our problem as the movement of the tile is sliding the entire row or column. We modified Manhattan distance  according to our problem requirement and converted every state in the  form of matrix to find the distance. As it was time consuming, we used  reshape function by importing  numpy package which improved the program  performance. Instead of just sorting the fringe and sending the state  with least cost to generate successor states, we generated the successor states and checked if it is visited. We checked if the cost is greater  than the cost from the current state and updated the state with  the  least cost. With misplaced numbers and Manhattan distance together as a  heuristic function, the program reached the goal state much faster. 
 Initial State - The given board. 
 Final State - Numbers arranged in an increasing order from 1 to 20. 
 Successor function - 18 possible states from any given states. 
 Heuristic function - Number of misplaced numbers + Manhattan distance from current position to goal position of every number.

Code Explanation -

1. The main function reads the board and calls solve function.
2. The initial board is added to the fringe, along with a field to keep track of route it takes to reach goal and cost from      its state to  goal state.
3. Initial board is popped and its successors are generated.
4. Until the fringe is empty, it generates the successor and find if has reached goal state.
5. Every time the fringe gets sorted based on cost and direction is updated along the path it takes to reach goal.
6. A dictionary is created to keep track of visited states along with the minimum cost required to reach the goal state.
7. For every state Manhattan distance is calculated.
8. If the state is already visited, the minimum cost is updated in the fringe for that particular state.
9. States which are not visited before are added to the fringe and popped to generate successors.
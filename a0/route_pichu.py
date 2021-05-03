#!/usr/local/bin/python3
#
# route_pichu.py : a maze solver
#
# Submitted by : [TUSHAR KANT SAMANTARAY, TSAMANT]
#
# Based on skeleton code provided in CSCI B551, Fall 2020.


import sys
import json

# Parse the map from a given filename
def parse_map(filename):
        with open(filename, "r") as f:
                return [[char for char in line] for line in f.read().rstrip("\n").split("\n")]
                
# Check if a row,col index pair is on the map
def valid_index(pos, n, m):
        return 0 <= pos[0] < n  and 0 <= pos[1] < m

# Find the possible moves from position (row, col)
def moves(map, row, col):
        moves=((row+1,col), (row-1,col), (row,col-1), (row,col+1))

	# Return only moves that are within the board and legal (i.e. go through open space ".")
        return [ move for move in moves if valid_index(move, len(map), len(map[0])) and (map[move[0]][move[1]] in ".@" ) ]

# Perform search on the map
def search1(house_map):
        # Find pichu start position
        pichu_loc=[(row_i,col_i) for col_i in range(len(house_map[0])) for row_i in range(len(house_map)) if house_map[row_i][col_i]=="p"][0]
        fringe=[(pichu_loc)]
        visited = [[False for col_i in range(len(house_map[0]))] for row_i in range(len(house_map))] # This is used to track all the visited nodes.
        track = [] # This is used to track the node paths travelled that will be used to map the shortest route
        while fringe:
                (curr_move)=fringe.pop()
                for move in moves(house_map, *curr_move):
                        if house_map[move[0]][move[1]]=="@":
                                track.append((curr_move, move))
                                directions = shortest_path(track, pichu_loc)
                                return len(directions), directions
                        else:
                                if visited[move[0]][move[1]] == False:
                                    visited[move[0]][move[1]] = True
                                    fringe.append(move)
                                    track.append((curr_move, move))
        return "Inf", "" #Return Inf is no path is found                            
# Find the shortest part from all the travesed paths. We travel from the goal to the starting point here, and use map_directions to map the path.
def shortest_path(track, pichu_loc):
    directions = []
    node = track.pop()
    map_direction(directions, node[1], node[0])
    last_node = node[0]
    while track:
        node = track.pop()
        if ((node[1][0]==pichu_loc[0] and node[1][1] == pichu_loc[1])):
            map_direction(directions, last_node, node)
            return
        if (node[1][0]==last_node[0] and node[1][1] == last_node[1]):
            map_direction(directions, last_node, node[0])
            last_node = node[0]
    directions.reverse()
    return directions

'''
 Map the direction travelled by the pichu in the shortest path. Since we are travelling backwards in the shortest_path function, that is,
 from the goal position to the position of the pichu, we have to track in compass direction in the opposite direction, and the reverse it
 to get the actual direction.
'''
def map_direction(directions, last_node, node):
    if (node[0] < last_node[0]):
        directions.append('S')
    elif (node[0] > last_node[0]):
        directions.append('N') 
    elif (node[1] < last_node[1]):
        directions.append('E')
    elif (node[1] > last_node[1]):
        directions.append('W')
    return directions

# Main Function
if __name__ == "__main__":
        house_map=parse_map('map3.txt')
        print("Shhhh... quiet while I navigate!")
        solution, directions = search1(house_map)
        print("Here's the solution I found:")
        print(solution, ''.join(map(str, directions)))

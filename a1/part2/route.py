#!/usr/local/bin/python3
# Code by: Tushar Kant Samantaray - tsamant, Monika Krishnamurthy - monkrish

import sys
import json

#NOTES
#road[0] - This is the current city
#road[1] - This is the next city
#road[2] - This is the distance between the current and next city
#road[3] - This is the speed limit of the road between the current and next city

#Find the bidirecitonal routes that are possible to travel to/from the current city
def routes(curr_city):
    return [(road[1], int(road[2]), travel_time(int(road[2]), int(road[3])), probability_of_accident(int(road[2]), int(road[3]))) for road in road_segments if road[0] == curr_city] + \
           [(road[0], int(road[2]), travel_time(int(road[2]), int(road[3])), probability_of_accident(int(road[2]), int(road[3]))) for road in road_segments if road[1] == curr_city]
 
# To calculate the probability of accident provided the distance and speed limit in a route   
def probability_of_accident(distance, speed):
    return 0.000001*(speed+5)*distance if cost_function == 'time' else 0.000001*speed*distance

# To calculate the time travelled provided the distance and speed limit in a route
def travel_time(distance, speed):
    return distance/(speed+5) if cost_function == 'time' else distance/speed


def find_path(city_gps, road_segments, cost):
    # The shortest path of current city contain the following information :
    # 1. It's parent city, 2. Segments travelled, 3. Distance travelled, 4. Time travelled, 5. Expected Accident
    shortest_path = {start_city: (None, 0, 0, 0, 0)}
    current_city = start_city
    visited_cities = set()
    
    while current_city != end_city:
        visited_cities.add(current_city)
        segments_travelled = shortest_path[current_city][1]
        distance_travelled = shortest_path[current_city][2]
        travel_time = shortest_path[current_city][3]
        probability_of_accident = shortest_path[current_city][4]
        
        for next_city in routes(current_city):
            segments = segments_travelled + 1
            distance = next_city[1] + distance_travelled
            time = next_city[2] + travel_time
            expected_accident = next_city[3] + probability_of_accident
            if next_city[0] not in shortest_path:
                shortest_path[next_city[0]] = (current_city, segments, distance, time, expected_accident)
            else:
                #Update the shortest path based on the cost function
                if cost == 'segments':
                    current_shortest_segments = shortest_path[next_city[0]][1]
                    if current_shortest_segments > segments:
                        shortest_path[next_city[0]] = (current_city, segments, distance, time, expected_accident)
                if cost == 'distance':
                    current_shortest_distance = shortest_path[next_city[0]][2]
                    if current_shortest_distance > distance:
                        shortest_path[next_city[0]] = (current_city, segments, distance, time, expected_accident)
                if cost == 'time':
                    current_shortest_time = shortest_path[next_city[0]][3]
                    if current_shortest_time > time:
                        shortest_path[next_city[0]] = (current_city, segments, distance, time, expected_accident)
                if cost == 'cycling':
                    current_shortest_expected_accident = shortest_path[next_city[0]][4]
                    if current_shortest_expected_accident > expected_accident:
                        shortest_path[next_city[0]] = (current_city, segments, distance, time, expected_accident)
        
        #Select the next unvisited city with the shortest path
        next_cities = {city: shortest_path[city] for city in shortest_path if city not in visited_cities}
        
        if not next_cities:
            return ("No routes are available from %s to %s" % (start_city, end_city))
        
        # Select the next minimum city based on the minimum value of the cost function
        if cost == 'segments':
            current_city = min(next_cities, key=lambda k: next_cities[k][1])
        if cost == 'distance':
            current_city = min(next_cities, key=lambda k: next_cities[k][2])
        if cost == 'time':
            current_city = min(next_cities, key=lambda k: next_cities[k][3])
        if cost == 'cycling':
            current_city = min(next_cities, key=lambda k: next_cities[k][4])
        
    # Find the route 
    route = []    
    while current_city is not None:
        route.append(current_city)
        if current_city == start_city:
            break
        next_city = shortest_path[current_city][0]
        current_city = next_city
    
    route = route[::-1]
    (total_segments, total_distance, total_time, total_expected_accident) = find_route_details(route)
    return (total_segments, total_distance, total_time, total_expected_accident, " ".join(route))

# Calculate all the details for the route
def find_route_details(route):
    total_distance = total_time = total_expected_accident = 0
    total_segments = len(route)-1
    for city1, city2 in zip(route[0::], route[1::]):
        for road in road_segments:
            if (road[0] == city1 and road[1] == city2) or (road[0] == city2 and road[1] == city1):
                total_distance += int(road[2])
                total_time += travel_time(int(road[2]), int(road[3]))
                total_expected_accident += probability_of_accident(int(road[2]), int(road[3]))                                                             
    return total_segments, total_distance, total_time, total_expected_accident

if __name__ == "__main__":
    if(len(sys.argv) != 4):
        raise(Exception("Error: Please provide all the required command line arguments"))

    city_gps = [] #To store all the city details provided
    road_segments = [] #To store all the road segment details provided    
    start_city = sys.argv[1]
    end_city = sys.argv[2]
    cost_function = sys.argv[3]
    
    #Read data from city-gps file
    with open('city-gps.txt') as file:
        city_gps = [ [ i.strip() for i in city_info.split(' ') ] for city_info in  file ]           
        file.close()
    #Read data from road-segments file
    with open('road-segments.txt') as file:
        road_segments = [ [ i.strip() for i in road_segment.split(' ') ] for road_segment in  file ]           
        file.close()
    
    # Check if the provided cities are valid or not
    if start_city not in [city[0] for city in city_gps]:
        raise(Exception("Error: Please provide a valid start city"))
    
    if end_city not in [city[0] for city in city_gps]:
        raise(Exception("Error: Please provide a valid end city"))

    # Check if the provided cost function is valid or not
    if cost_function not in ['segments', 'distance', 'time', 'cycling']:
        raise(Exception("Error: Please provide one of the following valid cost function: 1. segments 2. distance 3. time 4. cycling"))
    
    #Find the path based on the cost function
    solution = find_path(city_gps, road_segments, cost_function)
    print(*solution, ' ')
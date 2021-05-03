#!/usr/local/bin/python3

# put your group assignment program here!
import sys
import json  
import random

# Retruns the list of people perferred by a person
def preferred_people(person):
    return [person_preference[1] for person_preference in e_survey if person_preference[0] == person][0].split('-')

# Retruns the list of peple not perferred by a person
def not_preferred_people(person):
    return [person_preference[2] for person_preference in e_survey if person_preference[0] == person][0].split(',')

# Add a new team member
def add_new_member(current_team, new_member, minimum_cost, conflict_cost):
    current_team[1] = current_team[1] + minimum_cost
    current_team[2] = conflict_cost
    current_team[0].append(new_member)
    return 

# Find the requested team size
def requested_team_size(new_member):
    return len(preferred_people(new_member))
    
# Calculate the cost if a new member is added to a team
def calculate_new_cost(current_team, new_member):
    new_cost = 0
    size_conflict_cost = 0
    if (current_team != None):
        for member in current_team[0]:
            # Check if the new member does not prefer to work with a current member of the team
            if member in not_preferred_people(new_member):
                new_cost += m
            # Check if a current member of the team does not prefer to work with the new member
            if new_member in not_preferred_people(member):
                new_cost += m
            # Check if a current member is not in the preferred list of new member
            if member not in preferred_people(new_member) and 'zzz' not in preferred_people(new_member):
                new_cost += n
            # Check if the team size is as per the request of each current member
            if new_member in preferred_people(member):
                new_cost -= n
            if (current_team != None):
                if len(current_team[0])+1 != requested_team_size(member):
                    size_conflict_cost += 1  
        if len(current_team[0])+1 != requested_team_size(new_member):
            size_conflict_cost += 1 
    else:
        new_cost = n*(len(preferred_people(new_member)) - 1)
        # Check if the team size is as per the request of new member
        size_conflict_cost = 1 if (requested_team_size(new_member) != 1) else 0
    return new_cost, size_conflict_cost

# Calculate the total cost. When the value of 'new_team' is True, the function returns the total cost if a new team is formed
def total_cost(teams, new_team = False):
    total_cost = k*(len(teams)+1) if new_team == True else k*len(teams)
    for team in teams:
        total_cost += team[1]
        total_cost += team[2]
    return total_cost
    
# Call this function to form the teams
def form_teams(e_survey):
    teams = [] # Maintain a list of teams and their costs that are formed
    for person in e_survey:
        if len(teams) == 0:
            team_cost, size_conflict_cost = calculate_new_cost(None, person[0])
            teams.append([[person[0]], team_cost, size_conflict_cost])
        else:
            #Total cost of all the teams formed, along with conflict costs
            current_total_cost = total_cost(teams)
            #Toal cost of forming a new team
            cost_forming_new_team = total_cost(teams, True)
            #Cost of conflict of new member in a new team, conflict of new member not calculated yet
            team_cost, size_conflict_cost = calculate_new_cost(None, person[0])
            #Total cost of formming a new team along with the conflict cost in the new team
            total_cost_forming_new_team = cost_forming_new_team + team_cost + size_conflict_cost
            # Cost of adding the new member to an existing team
            minimum_cost_team = None
            minimum_cost = total_cost_forming_new_team
            for team in teams:                
                if len(team[0]) < 3:
                    team_cost = team[1] + team[2]
                    new_cost, new_conflict_cost = calculate_new_cost(team, person[0])
                    if (current_total_cost - team_cost) + new_cost + new_conflict_cost < minimum_cost:
                        minimum_cost_team = team
                        minimum_cost = (current_total_cost - team_cost) + new_cost + new_conflict_cost
            # If formning a new team costs less than adding the new memberto any of the existing team
            if minimum_cost_team == None:
                teams.append([[person[0]], team_cost, size_conflict_cost])
            else:
                for team in teams:
                    if team == minimum_cost_team:
                        team_cost, size_conflict_cost = calculate_new_cost(team, person[0])
                        add_new_member(minimum_cost_team, person[0], team_cost, size_conflict_cost)    
    return teams, total_cost(teams)

if __name__ == "__main__":
    if(len(sys.argv) != 5):
        raise(Exception("Error: Please provide all the required command line arguments"))
    
    e_survey = [] 
    k = int(sys.argv[2])
    m = int(sys.argv[3])
    n = int(sys.argv[4])

    #Read data from e_survey file
    with open(sys.argv[1]) as file:
        e_survey = [ [ i.strip() for i in response.split(' ') ] for response in file ]           
        file.close()
    
    # Track the previous minimum cost. Initialised with a very high value
    previous_cost = 100000
    checked_arrangements = []
    # Find the combination of team and the total cost incurred
    for x in range(len(e_survey)**len(e_survey)):
        random.shuffle(e_survey)
        if e_survey not in checked_arrangements:
            checked_arrangements.append([e_survey])
            (formed_teams, cost) = form_teams(e_survey)
            if cost < previous_cost:
                previous_cost = cost
                for team in formed_teams:
                    print(*team[0], sep='-')
                print(cost)
## Approach – <br/>

1.	Sebastian is one-player game. The program runs 100 times, and in every game, 5 dices are rolled to fill as much as 13 categories. Three chances are given to the player to inspect the dice and choose any subset and roll them.
2.	We first tried to understand the game. We started by reading the rules of the game and tried playing it ourself to understand how we would plan to choose a category or reroll some or all dice. 
3.	Based on our understanding of the game and the strategy we used, we devised the program to maximize the score. The strategy we used is explained below.  
4.	The Basic strategy is to focus on bonus from the Upper Sections. Our attempt to focus on the Upper category was inspired by the article, Bonus-Focused Yatzy.
5.	We prioritized selecting the highest scoring categories to maximize the score. 

## Sebastian Auto-Player strategy - <br/>

1.	The main program, Sebastian.py calls SebastianAutoPlayer.py 3 times to roll 5 dices. 
2.	A class called HelperFunction is created with functions such as, <br/>
i.	Available_categoires – Helps to keep track of categories available for assigning the dices. <br/>
ii.	Dice_roll – Helps to keep the dices in list format. <br/>
iii.	Most_frequent – Tracks more frequently appeared number in a roll. <br/>
iv.	Value_counts – Counts the face value of the dices in the roll. <br/>
v.	Comb – Shows all possible combinations of dice. <br/>
vi.	Choose_dice_reroll – Helps to identify the dices that should be rerolled. <br/>
3.	When the dices are rolled, with the help of value_counts, the score of the rolls is calculated. The dices are checked to see if it can be allotted to any of the 13 categories. 
4.	Priority is given to the high scoring categories. 
5.	If all the dices are same the program would assign the dice roll to ‘Quintuplicatam’ and the program would decide not to reroll.
6.	If the dices roll either ‘1,2,3,4,5’ or ‘2,3,4,5,6’ the program would assign the roll to ‘Company’ because the probability of rolling a Quintuplicatam from a Company in 2 rolls is very less.
7.	If the dice roll matches a ‘Prattle’, the program first check if ‘Company’ is still available and if available the program would choose the 5th dice to make it a Company. If Company is not attained in next 2 rolls, the program will allocate the category to ‘Prattle’.
8.	If the dice rolls four of the same number, the program would try to roll the 5th dice to get ‘Quintuplicatam’ in 2 rolls if the category is still available. If it is not available or does not roll five of the same kind, the program will assign the roll to ‘Quadrupla’.
9.	If the dice rolls three dice of one number and two dice of another number, the program checks if ‘Quadrupla’ or ‘Quintuplicatam’ is available. If not available, the program will assign the roll to ‘Squadron’. If available, the program will set aside the three of the kind and rerolls the other two dice to attain them. If one of the two is attained, the program will assign the roll to the respective category or else the program will assign the roll to ‘Triplex’. If the two dice are a pair again after the two rolls, the program will assign the roll to ‘Squadron’.
10.	If the dice rolls three dice of one number, the program checks if ‘Squadron’ or ‘Quadrupla’ or ‘Quintuplicatam’ is available. If not available, the program will assign the roll to ‘Triplex’. If at least one is available, the program will set aside the three of the kind and rerolls the remaining two dice to attain it. If a higher scoring category is attained, the program will assign the roll to the respective category or else the program will assign the roll to ‘Triplex’. 
11.	If none of the above categories are matched, the program will set aside the most frequent dice and reroll the remaining to match the any of the Upper Category.
12.	If none of the categories are available, the program will assign the roll to Pandemonium.
13.	The score is updated after each assignment to a category. After 13 rolls, the program will display the final score.
14.	The program will continue to run the same 100 times, and the program will display the minimum, maximum and mean of the 100 iterations. 

Reference:
1 - Bonus-Focused Yatzy - A COMPARISON BETWEEN AN OPTIMAL STRATEGY AND A BONUS-FOCUSED STRATEGY - DARJA HASHEMI DEMNEH 
http://www.diva-portal.org/smash/get/diva2:812196/FULLTEXT01.pdf <br/>
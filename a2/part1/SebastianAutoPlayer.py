# Automatic Sebastian game player
# B551 Fall 2020
# PUT YOUR NAME AND USER ID HERE!
#
# Based on skeleton code by D. Crandall
#
#
# This is the file you should modify to create your new smart player.
# The main program calls this program three times for each turn. 
#   1. First it calls first_roll, passing in a Dice object which records the
#      result of the first roll (state of 5 dice) and current Scorecard.
#      You should implement this method so that it returns a (0-based) list 
#      of dice indices that should be re-rolled.
#   
#   2. It then re-rolls the specified dice, and calls second_roll, with
#      the new state of the dice and scorecard. This method should also return
#      a list of dice indices that should be re-rolled.
#
#   3. Finally it calls third_roll, with the final state of the dice.
#      This function should return the name of a scorecard category that 
#      this roll should be recorded under. The names of the scorecard entries
#      are given in Scorecard.Categories.
#

from SebastianState import Dice
from SebastianState import Scorecard
import random
import numpy as np
from itertools import combinations 

class HelperFunctions:
    re_roll = []
    # Returns a list of available categoeis to be filled
    def available_categories(scorecard):
        return list(set(Scorecard.Categories) - set(scorecard.scorecard.keys())) 
    # Returns the die rolls as a list
    def dice_rolls(dice):
        return [int(roll) for roll in list((str(dice).split(" ")))]
    # Returns the most frequently rolled value in all dices
    def most_frequent(rolls): 
        return max(set(rolls), key = rolls.count), rolls.count(max(set(rolls), key = rolls.count))
    # Returns count of occurance of each value in all dices
    def value_counts(rolls):
        return [rolls.count(i) for i in range(1,7)]
    # Pick all possible combinations of subset of n size as 'list of tuples'
    def comb(rolls, n):
        return list(combinations(rolls, n))
    # Returns the dices that should be rerolled
    def choose_dice_reroll(rolls, counts, available_categories):
        # Don't re-roll if we have a high scoring group available
        # Check if we have Quintuplicatam
        if ('quintuplicatam' in available_categories and all(roll == rolls[0] for roll in rolls)):
            return []
        elif ('company' in available_categories and (sorted(rolls) == [1,2,3,4,5] or sorted(rolls) == [2,3,4,5,6])):
            return []
        elif ('prattle' in available_categories and (len(set([1,2,3,4]) - set(rolls)) == 0 or len(set([2,3,4,5]) - set(rolls)) == 0 or len(set([3,4,5,6]) - set(rolls)) == 0)):

            if ('company' in available_categories): # Try to get a company if it is available
                largest_most_frequent_count = max(counts[:-1]) # Find the frequency of the most repeated value
                largest_most_frequent_roll = counts.index(largest_most_frequent_count)+1 #Find the value with the highest frequency
                if largest_most_frequent_count > 1:
                    for i in range(0, 5):
                        if rolls[i] == largest_most_frequent_roll: # Re-roll the repeated value
                            return [i]
                    else:
                        return [rolls.index(6)]

            return [] # Do not re-roll
        elif ('quadrupla' in available_categories and (4 in counts)):
            largest_most_frequent_roll = counts.index(max(counts[:-1]))+1
            for i in range(0, 5): # Iterate over all 5 dice rolls
                if rolls[i] != largest_most_frequent_roll: # If a four-of-a-type subset is available, re-roll the last dice if quintuplicatam is available, or if the value of the last dice is less than 3
                    return [i] if ('quintuplicatam' in available_categories or rolls[i] < 6) else []

        elif ('squadron' in available_categories and (2 in counts) and (3 in counts)):
            if ('quadrupla' in available_categories or 'quintuplicatam' in available_categories): # Try to get a 4-of-a-kind quadrupa or 5-of-a-kind quintuplicatam if they are still not filled yet
                largest_most_frequent_count = max(counts[:-1]) # Find the frequency of the most repeated value
                largest_most_frequent_roll = counts.index(largest_most_frequent_count)+1 #Find the value with the highest frequency
                x = []
                for i in range(0, 5): # Iterate over all 5 dice rolls
                    if rolls[i] != largest_most_frequent_roll:
                        x.append(i)
                HelperFunctions.re_roll = x
                return HelperFunctions.re_roll
            return [] # Do not re-roll
        elif ('triplex' in available_categories and (3 in counts)):
            if ('squadron' in available_categories or 'quadrupla' in available_categories or 'quintuplicatam' in available_categories): # Try to get a 4-of-a-kind quadrupa or 5-of-a-kind quintuplicatam if they are still not filled yet
                largest_most_frequent_count = max(counts[:-1]) # Find the frequency of the most repeated value
                largest_most_frequent_roll = counts.index(largest_most_frequent_count)+1 #Find the value with the highest frequency
                x = []
                for i in range(0, 5): # Iterate over all 5 dice rolls
                    if rolls[i] != largest_most_frequent_roll:
                        x.append(i)
                HelperFunctions.re_roll = x
                return HelperFunctions.re_roll
            return [] # Do not re-roll
        else:
            # Re-roll all the dices except the highest frequency ones to maximize counts
            largest_most_frequent_count = max(counts[:-1]) # Find the frequency of the most repeated value
            largest_most_frequent_roll = counts.index(largest_most_frequent_count)+1 #Find the value with the highest frequency
            x = []
            for i in range(0,5):
                if rolls[i] != largest_most_frequent_roll:
                    x.append(i)
            HelperFunctions.re_roll = x
            return HelperFunctions.re_roll
class SebastianAutoPlayer:
    group_priorities = {'quintuplicatam': 1, 'pandemonium': 2, 'company': 3, 'squadron': 4,
    'quadrupla': 5, 'triplex': 6}

    def __init__(self):
        pass

    def first_roll(self, dice, scorecard):
        available_categories = HelperFunctions.available_categories(scorecard)
        rolls = HelperFunctions.dice_rolls(dice)
        counts = HelperFunctions.value_counts(rolls)
        return HelperFunctions.choose_dice_reroll(rolls, counts, available_categories)              

    def second_roll(self, dice, scorecard):
        available_categories = HelperFunctions.available_categories(scorecard)
        rolls = HelperFunctions.dice_rolls(dice)
        counts = HelperFunctions.value_counts(rolls)
        return HelperFunctions.choose_dice_reroll(rolls, counts, available_categories)   
      
    def third_roll(self, dice, scorecard):
        chosen_category = ''
        available_categories = HelperFunctions.available_categories(scorecard)
        rolls = HelperFunctions.dice_rolls(dice)
        counts = HelperFunctions.value_counts(rolls)
        # Prioritize choosing high scoring groups first
        if (chosen_category == '' and 'quintuplicatam' in available_categories and all(roll == rolls[0] for roll in rolls)):
            chosen_category = 'quintuplicatam'
        if (chosen_category == '' and 'company' in available_categories and (sorted(rolls) == [1,2,3,4,5] or sorted(rolls) == [2,3,4,5,6])):
            chosen_category = 'company'
        if (chosen_category == '' and 'prattle' in available_categories and (len(set([1,2,3,4]) - set(rolls)) == 0 or len(set([2,3,4,5]) - set(rolls)) == 0 or len(set([3,4,5,6]) - set(rolls)) == 0)):
            chosen_category = 'prattle'
        if (chosen_category == '' and 'quadrupla' in available_categories):
            for n in range(1, 7):
                if ((n, n, n, n) in HelperFunctions.comb(rolls, 4)):
                    chosen_category = 'quadrupla'
        if (chosen_category == '' and 'squadron' in available_categories and (2 in counts) and (3 in counts)):
            chosen_category = 'squadron'
        if (chosen_category == '' and 'triplex' in available_categories and (3 in counts)):
            chosen_category = 'triplex'
        else:
            largest_most_frequent_roll = counts.index(max(counts[:-1]))+1
            if (chosen_category == '' and largest_most_frequent_roll == 6 and 'sextus' in available_categories):
                chosen_category = 'sextus'
            elif (chosen_category == '' and largest_most_frequent_roll == 5 and 'quintus' in available_categories):
                chosen_category = 'quintus'
            elif (chosen_category == '' and largest_most_frequent_roll == 4 and 'quartus' in available_categories):
                chosen_category = 'quartus'
            elif (chosen_category == '' and largest_most_frequent_roll == 3 and 'tertium' in available_categories):
                chosen_category = 'tertium'
            elif (chosen_category == '' and largest_most_frequent_roll == 2 and 'secundus' in available_categories):
                chosen_category = 'secundus'
            elif (chosen_category == '' and largest_most_frequent_roll == 1 and 'primis' in available_categories):
                chosen_category = 'primis'
            else:
                if (chosen_category == ''):
                    chosen_category = 'pandemonium'
        return chosen_category







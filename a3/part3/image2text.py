#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Tushar Kant Samantaray: tsamant, Monika Krishnamurthy: monkrish, Anubhav Lavania: alavania
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import string
import math
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' " # Moved it to top so this can be referenced in other functions

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# Use the bc.train file to find the initial and transition probabilites of each character in the bc.train file
def train(train_txt_fname):
    # Prepare words and letters set
    words_train_data = prepare_training_data(train_txt_fname)

    # Calculate transition probability between two letters and the initial probability of starting a word with any letter
    character_transition, character_initial = create_character_dict(words_train_data)

    return character_initial, character_transition

# Create two dictionary of initial characters and transition probability between all possible combination of characters
def create_character_dict(words_train_data):
    character_transition = {} # To store the transition probabilities from one character to the other in the format {key = 'char1char2', value = count/total}
    character_initial = {} # To store the initial probablity of a character in the format {key = 'char', value = count/total}
    total_characters = 0 # Count the total characters to normalise

    for word in words_train_data:
        initial_character = word[0] # Track the initial character to calculate the initial probability of each character
        total_characters += len(word) - 1 # -1 to remove added space
        if initial_character not in character_initial: # For initial characters
            character_initial[initial_character] = 1
        else:
            character_initial[initial_character] += 1
        for i in range(len(word) - 1): # For transitions
            char_i = word[i]
            char_j = word[i+1]

            key = char_i + char_j # Store the count for transition from character i to character j
            if key in character_transition:
                character_transition[key] += 1
            else:
                character_transition[key] = 1

    # Handle missing trainsition and initalisations
    for character in TRAIN_LETTERS:
        if character not in character_initial.keys():
            character_initial[character] = 100
        for character2 in TRAIN_LETTERS:
            key = character + character2
            if key not in character_transition.keys():
                character_transition[key] = 100 # By experimenting, giving a higher value of 100 to missing transitions instead of very small value like 1 gives better outputs

    # Normalise the transition and initalisations
    for key in character_transition.keys():
        character_transition[key] = character_transition[key]/total_characters    
    for key in character_initial.keys():
        character_initial[key] = character_initial[key]/len(words_train_data)

    return (character_transition, character_initial)

# Read the bc.train file and remove the POS. Also add a space at the end of each word as we need to compute the transition from a character to space.
def prepare_training_data(train_txt_fname):
    words = []
    
    file = open(train_txt_fname, 'r')
    for line in file:   
        sentence = remove_pos(line)
        for word in sentence:
            words.append(word)
    return words

# Remove POS from bc.train datafile and add a space at the end of each word to compute transition probabilities between each characters
def remove_pos(sentence):
    sentence_without_pos = []
    POS = ['DET', 'NOUN', 'ADJ', 'VERB', 'PRT', 'ADP', 'NUM', 'ADV', 'CONJ', 'PRON', 'PRT']
    for word in sentence.split():
        if word not in POS:
            sentence_without_pos.append(word + ' ')
    return sentence_without_pos

'''
Find the maximum emission probability for a sequence of test letters against the train letters and returns the indeces of
the most sequence of characters with highest emission probabilities
'''
def emission_probability(train_letters, test_letters):
    noise = 0.28 # By experimenting with different noise values, a value of 0.3 gives us better results
    emission_probability = np.zeros(shape = (len(train_letters), len(test_letters)))

    for i, train_letter in enumerate(train_letters): # Enumerate over each character from training set
        train_letter_pixels = train_letters.get(train_letter) 
        for j, test_letter_pixels in enumerate(test_letters): # Enumerate over each character from test test
            matched, not_matched = 0, 0
            for test_pixel_row, train_pixel_row in zip(test_letter_pixels, train_letter_pixels): # Iterate over each rows of training and test pixels
                for test_pixel, train_pixel in zip(test_pixel_row, train_pixel_row): # Compare each pixel of training and test character
                    if test_pixel == train_pixel: 
                        matched += 2 if test_pixel == ' ' else 1 # Giving higher score for active text pixels over blank spaces gives significantly better results
                    else:
                        not_matched += 1 if test_pixel == ' ' else 2 # Giving lower score to unmatached active pixels over blank spaces gives significantly better results
            emission_probability[i][j] = (math.pow(noise, not_matched)) * (math.pow(1 - noise, matched)) # Store the computed emission probabilities
    return emission_probability

# Predict the sentence of test letters just using the emission probability
def simplified(train_letters, test_letters):
    predicted_indices = np.zeros(len(test_letters)) # Create a nparray with the size of test letters to store the indeces of the maximum likely characters 
    predicted_sentence = ''
    
    emission = emission_probability(train_letters, test_letters) # Find the emission probabilities for each character in test_letters
    predicted_indices = np.argmax(emission, axis=0) # Find the maximum probability of the computed emission probabilities

    for i in predicted_indices:
        predicted_sentence += TRAIN_LETTERS[i] #Get the actual character based on the predicted indeces
    return predicted_sentence

def hmm_viterbi(train_letters, test_letters, character_initial, character_transition):
    emission = emission_probability(train_letters, test_letters) # Fetch the emission probabilities
    predicted_sentence = [] # This list holds the predicted characters of the test_letters

    for i, test_letter in enumerate(test_letters):
        p = {} # This dictionary is used to hold the list of calculated probabilities of each train_letter against a test_letter.
        for j, train_letter in enumerate(train_letters):
            if i == 0: # Get the initial probability for the first character
                transition_p = character_initial[train_letter]
            else: # Else get the transition probability
                key = predicted_sentence[-1] + train_letter
                transition_p = character_transition[key]
            emission_p = emission[j, i]
            character_p = emission_p * transition_p # Compute the probability of getting a character
            p[train_letter] = character_p # Store it in p dictionary from which we need to select the character with maximum probability
        
        predict_max = max(p, key=p.get) # Pick the character with the maximum computed probability
        predicted_sentence.append(predict_max)
    return "".join(predicted_sentence)

#####
# main program

# For testing
# (train_img_fname, train_txt_fname, test_img_fname) = 'courier-train.png', 'bc.train', 'test-15-0.png'

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!

# Train the model and compute the initial and transition probabilitie of the characters
character_initial, character_transition = train(train_txt_fname)

def solve(algo, train_letters, test_letters, character_initial, character_transition):
    if algo == "Simple":
        return simplified(train_letters, test_letters)
    elif algo == "HMM":
        return hmm_viterbi(train_letters, test_letters, character_initial, character_transition)
    else:
        print("Unknown algo!")

Algorithms = ("Simple", "HMM")
outputs = {}

# Solve for each algorithms
for algo in Algorithms:
    outputs[algo] = solve(algo, train_letters, test_letters, character_initial, character_transition)

# The final two lines of your output should look something like this:
print("Simple: " + outputs["Simple"])
print("   HMM: " + outputs["HMM"]) 
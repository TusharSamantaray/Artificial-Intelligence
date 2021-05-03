#!/usr/local/bin/python3
# CSCI B551 Fall 2020
#
# Authors: Tushar Kant Samantaray: tsamant, Monika Krishnamurthy: monkrish, Anubhav Lavania: alavania
#
# based on skeleton code by D. Crandall, 11/2020
#
# ./break_code.py : attack encryption
#


import random
import math
import copy 
import sys
import encode
import string
import datetime
import copy
import numpy as np
# put your code here!

# Create a dictionary from a decryption replacement guess that maps each alphabet to it's replacement alphabet
def create_replacement_dict(replacement_guess):
    replacement_dict = {}
    alphabet_list = list(string.ascii_lowercase)
    for i in range(len(replacement_guess)):
        replacement_dict[alphabet_list[i]] = replacement_guess[i]
    return replacement_dict

# Decrpyt a document based on a randomly guessed replacement and rearrangement order
def decrypt_document(encrypted_document, replacement_guess, rearrangement_guess):
    # Apply the rearrangment guess on the current encrypted document
    encrypted_document = rearrange_encrypted_document(encrypted_document, rearrangement_guess)

    # Apply the replacement guess on the current encrypted document
    new_document = ""
    document = list(encrypted_document)
    replacement_dict = create_replacement_dict(replacement_guess)

    for alphabet in document:
        if alphabet.isalpha() and alphabet.lower() in replacement_dict:
            new_document += replacement_dict[alphabet.lower()]
        else:
            new_document += " "
    return new_document

'''
Create a dictionary of transition from one alphabet to the other for a given document.
Both corpus and decrpted documents based on rearrangement and replacement guesses use this function to calculate the transitions.
If the total count from transition from alphabet a to b is 300, the dictionary holds the value in the format {'ab': count}
'''
def create_alphabet_transition_dict(document):
    alphabet_transition = {}
    alphabet_list = list(string.ascii_lowercase)
    total_alphabets = 1

    for i in range(len(document) - 1):
        alpha_i = document[i].lower() if document[i].isalpha() else " "
        alpha_j = document[i+1].lower() if document[i+1].isalpha() else " "

        key = alpha_i + alpha_j
        if key in alphabet_transition:
            alphabet_transition[key] += 1
        else:
            alphabet_transition[key] = 1
        total_alphabets += 1

    # Handle missing data
    for alpha_i in alphabet_list:
        for alpha_j in alphabet_list:
            key = alpha_i + alpha_j
            if key not in alphabet_list:
                alphabet_transition[key] = 1 # Add a value of 100 for missing data, they'll get normalised below
    '''
    # To check probabilities of transition from alphabet1 -> alphabet2 -> alphabet3. This didn't gave us any significant improvement, but rather increased the execution time.
    for i in range(len(document) - 2):
        alpha_i = document[i].lower() if document[i].isalpha() else " "
        alpha_j = document[i+1].lower() if document[i+1].isalpha() else " "
        alpha_k = document[i+2].lower() if document[i+2].isalpha() else " "

        key = alpha_i + alpha_j + alpha_k
        if key in alphabet_transition:
            alphabet_transition[key] += 1
        else:
            alphabet_transition[key] = 1
    '''
    # Normalise the alphabet_transition dictionary
    for key, value in alphabet_transition.items():
        alphabet_transition[key] = value/total_alphabets
    
    return alphabet_transition

# Calculate the score of a document based on randomly guessed replacement and rearrangement order
def get_decrypted_document_score(document, alphabet_transition, replacement_guess, rearrangement_guess):
    decrypted_document = decrypt_document(document, replacement_guess, rearrangement_guess)
    decrypted_document_transition = create_alphabet_transition_dict(decrypted_document)
    decrypted_document_score = 0

    for key, value in decrypted_document_transition.items():
        if key in alphabet_transition:
            decrypted_document_score += value * math.log(alphabet_transition[key]) 
    return decrypted_document_score

# Creates a random rearrangement guess for alphabets order
def generate_new_rearrangement_guess(current_arrangement):
    new_arrangement = copy.deepcopy(current_arrangement)
    while new_arrangement == current_arrangement:
        random.shuffle(new_arrangement)
    return new_arrangement

# Create a new random replacement guess by swapping two alpabets
def generate_new_replacement_guess(current_replacement_guess):
    curr = list(current_replacement_guess)
    randomIndexes = random.sample(range(0, 26), 2) # Find two random indexes between 0-25
    while all(index == randomIndexes[0] for index in randomIndexes): # Find random indexes again if the found indexes are same
        randomIndexes = random.sample(range(0, 26), 2)
    curr[randomIndexes[0]], curr[randomIndexes[1]] = curr[randomIndexes[1]], curr[randomIndexes[0]] # Swap the characters of two randomly selected indexes

    return "".join(curr)    

# Rearragne the document based on a randomly generated rearrangement order
def rearrange_encrypted_document(encrypted_document, order):
    n = 4
    rearranged_document = []
    encrypted_document = [(encrypted_document[i:i+n]) for i in range(0, len(encrypted_document), n)]
    for chunk in encrypted_document:
        chunk = list(chunk)
        if len(chunk) < n:
            for i in range(n - len(chunk)): # Add blank space if the last chunk is not of size n
                chunk.append(' ')
        chunk[order[0]], chunk[order[1]], chunk[order[2]], chunk[order[3]] = chunk[0], chunk[1], chunk[2], chunk[3]
        rearranged_document.append("".join(chunk))
    return "".join(rearranged_document)

# Accept a guessed replacement and rearrangement with probability less than exp(new-current)
# Return True if the exponential is uncomputable
def accept_guess_score(score_new_guess, score_current_guess, count):
    p = (count+1)/30 # A variable scarling parameter to handle local minimum issue. The scaling parameter increaes as the iteration increases
    #return (random.uniform(0,1)) < score_new_guess/score_current_guess
    try:   
        return (random.uniform(0,1)) < math.exp(score_new_guess - score_current_guess)
    except: # As math.exp might return 'math range error'
        return True if score_new_guess < score_current_guess else False # If the new score is better than the previous one, accept the new arrangement.

def break_code(encrypred_document, corpus):
    start = datetime.datetime.now()
    end = start + datetime.timedelta(0, 595) # To terminate the program within 10 minutes
    #Iterations = 50000 # For testing

    # Initial replacement and rearrangemnt guess
    current_replacement_guess = string.ascii_lowercase
    current_rearrangement_guess = [0, 1, 2, 3]
    
    # To Maintain rearrangement and replacement guess and their respective scores
    best_replacement_guess, best_rearrangement_guess, best_score = '', '', 0

    # Create an alphabet transition dicitonary from given corpus
    alphabet_transition = create_alphabet_transition_dict(corpus) 
    count = 0 # Use this counter to print the decrypted document after every 1000 iterations, and also to calculate scaling parameter p
    while end > datetime.datetime.now():
        count += 1
        
        # Generate new guesses
        new_replacement_guess = generate_new_replacement_guess(current_replacement_guess)
        new_rearrangement_guess = generate_new_rearrangement_guess(current_rearrangement_guess)

        score_current_guess = get_decrypted_document_score(encrypred_document, alphabet_transition, current_replacement_guess, current_rearrangement_guess)
        score_new_guess = get_decrypted_document_score(encrypred_document, alphabet_transition, new_replacement_guess, new_rearrangement_guess)

        # Update the best guess if it's score is higher than the previous best guess
        if score_new_guess < best_score:
            best_replacement_guess = new_replacement_guess
            best_rearrangement_guess = new_rearrangement_guess
            best_score = score_new_guess

        # Accept the new guessed replacement and rearrangement sequences based on random chance
        if accept_guess_score(score_new_guess, score_current_guess, count):
            current_replacement_guess = new_replacement_guess
            current_rearrangement_guess = new_rearrangement_guess


        #For testing
        if count % 1000 == 0: # Print best decrypted document yet
            print("Best decrpytion yet :", + best_score)
            print(decrypt_document(encrypred_document, best_replacement_guess, best_rearrangement_guess))
        '''
        if count == Iterations:
            exit()
        '''
    return decrypt_document(encrypred_document, best_replacement_guess, best_rearrangement_guess)

if __name__== "__main__":
    
    if(len(sys.argv) != 4):
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = encode.read_clean_file(sys.argv[1])
    corpus = encode.read_clean_file(sys.argv[2])
    decoded = break_code(encoded, corpus)

    '''
    # For testing
    encoded = encode.read_clean_file("encrypted-text-1.txt")
    corpus = encode.read_clean_file("corpus.txt")
    '''
    decoded = break_code(encoded, corpus)

    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)

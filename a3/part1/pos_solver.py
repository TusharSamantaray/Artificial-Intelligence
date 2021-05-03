###################################
# CS B551 Fall 2020, Assignment #3
#
# Authors: Tushar Kant Samantaray: tsamant, Monika Krishnamurthy: monkrish, Anubhav Lavania: alavania
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np
import pandas as pd
from math import log

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
labels_words = [] # Store tuples of all (word, label) available in the training dataset
computed_emission = {} # Use this dictionary to save any computed emission probabilities, which we can resue for future lookups

class Solver:
    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        posterior_prob = 0
        if model == "Simple":
            for i, (l, w) in enumerate(zip(label, sentence)):
                if w+l in computed_emission:
                    emission = computed_emission[w+l]
                else:
                    emission = self.emission_probability(w, l)
                    computed_emission[w+l] = emission
                posterior_prob += 0.000001 if emission == 0 else log(emission) # Find log of emission probability for simple model
            return posterior_prob

        elif model == "HMM":
            for i, (l, w) in enumerate(zip(label, sentence)):
                if w+l in computed_emission:
                    emission = computed_emission[w+l]
                else:
                    emission = self.emission_probability(w, l)
                    computed_emission[w+l] = emission
                posterior_prob += 0.000001 if emission == 0 else log(emission) # Find log of emission probaiblity

                if i == 0: # For the initial probability, look at the probability from '.' to the first POS in the computed labels dataframe
                    transition_p = labels_df.loc['.', label[i]]
                else:
                    transition_p = labels_df.loc[label[i-1], label[i]] 
                posterior_prob += 0.000001 if transition_p == 0 else log(transition_p) # Add log of transition probability
            return posterior_prob

        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        for sentence in data:
            #print("sentence is", sentence)
            words = sentence[:1][0] # Each word in the sentence
            #print("words are", words)
            labels = sentence[1:][0] # Each label in the sentence

            for index in range(0, len(words)):
                labels_words.append((words[index], labels[index]))
        
        # Use set datatype to check how many unique tags are available in the training dataset
        labels = {label for word, label in labels_words}
        words = {word for word, label in labels_words}
        
        # Create a 2D matrix to store the emission probabilities of each available label
        labels_matrix = np.zeros((len(labels), len(labels)), dtype='float32')
        for i, l1 in enumerate(list(labels)):
            for j, l2 in enumerate(list(labels)):
                labels_matrix[i, j] = self.transition_probability(l2, l1)
        # Convert the matrix to a dataframe for easier access during probability calculations
        global labels_df # Use a global dataframe to store the computed probabilities and use it later for calculations
        labels_df = pd.DataFrame(labels_matrix, columns = list(labels), index = list(labels))
        #print(labels_df)

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        states = [] # Maintain the predicted set of states based on probabiity calculations
        labels = list(set([pair[1] for pair in labels_words]))
        for index, word in enumerate(sentence):
            p = [] # Used to maintain a list of computed probabilities for all labels
            for label in labels:
                emission_p = self.emission_probability(sentence[index], label)
                p.append(emission_p)

            state_max = labels[p.index(max(p))] # Find the state with maximum probability
            states.append(state_max) # Add the maxium state to the states list
        return states

    def hmm_viterbi(self, sentence):
        states = [] # Maintain the predicted set of states based on probabiity calculations
        labels = list(set([pair[1] for pair in labels_words]))
        for index, word in enumerate(sentence):
            #initialist list of probability column for a given observation
            p = [] # Used to maintain a list of computed probabilities for all labels
            for label in labels:
                if index == 0: # Check if it is the first label
                    transition_p = labels_df.loc['.', label]
                else:
                    transition_p = labels_df.loc[states[-1], label] # Lookup the transition probability in the pre-computed labels dataframe
                    #print(transition_p)
                if sentence[index]+label in computed_emission:
                    emission_p = computed_emission[sentence[index]+label ]
                else:
                    emission_p = self.emission_probability(sentence[index], label)
                    computed_emission[sentence[index]+label] = emission_p
                state_p = emission_p * transition_p # Calculate the state probability from emission and transition probability
                p.append(state_p)

            state_max = labels[p.index(max(p))] # Find the state with maximum probability
            states.append(state_max)
        return states
    '''
    In some cases, we get the condifence value as 'nan' with the following error  RuntimeWarning: invalid value encountered in double_scalars
    posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states}). We were not able to figure out how to solve this issue and decided
    to leave it for the moment.
    '''
    def confidence(self, sentence, answer):
        observation = sentence # The observations are the words of the sentence
        states = answer # The states are the computed POS

        transition_probability = labels_df # Renaming for better readability

        # Forward Part
        fwd = [] # To store teh forward probabilities
        for i, ob in enumerate(observation):
            f_curr = {}
            for st in states:
                if i == 0:
                    prev_f_sum = transition_probability.loc['.', st]
                else:
                    prev_f_sum = sum(f_prev[k] * transition_probability.loc[k, st] for k in states)
                if observation[i]+st in computed_emission:
                    emission_p = computed_emission[observation[i]+st]
                else:
                    emission_p = self.emission_probability(observation[i], st)
                    computed_emission[observation[i]+st] = emission_p
                f_curr[st] = emission_p * prev_f_sum
            fwd.append(f_curr)
            f_prev = f_curr
        p_fwd = sum(f_curr[k] * transition_probability.loc['.', k] for k in states)

        # Backward part
        bkw = []
        for i, ob in enumerate(observation[::-1]):
            b_curr = {}
            for st in states:
                if i == 0:
                    b_curr[st] = transition_probability.loc[st, '.']
                else:
                    b_curr[st] = sum(transition_probability.loc[st][l] * (computed_emission[observation[i]+l] if observation[i]+l in computed_emission else self.emission_probability(observation[i], l)) * b_prev[l] for l in states)            
            bkw.insert(0, b_curr)
            b_prev = b_curr

        p_bkw = sum(transition_probability.loc['.', l] * computed_emission[observation[0]+l] if observation[0]+l in computed_emission else self.emission_probability(observation[0], l) * b_curr[l] for l in states)

        # Merging forward and backward
        posterior = []
        for i in range(len(observation)):
            posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})
        
        # Compute the confidence of the state with highest computed probability
        confidence = []
        for row in posterior:
            total = 0
            maximum = 0
            for val in row.values():
                total = total + val
                if val > maximum:
                    maximum = val
            confidence.append(round(maximum/total, 2))
        return confidence
    
    # Calculate emission probability
    def emission_probability(self, word, label):
        label_list = [pair for pair in labels_words if pair[1] == label] 
        count_label = len(label_list) # Total number of times a tag appears in the training dataset
        
        word_label_list = [pair for pair in label_list if pair[0] == word]
        count_word_label_list = len(word_label_list) # Total number of times a given word occurs as a given tag

        return count_word_label_list/count_label # Total number of times a word was found for a given label/Total number of counts for the given label in the dataset

    # Calculate transition probability
    def transition_probability(self, label2, label1):
        labels = [pair[1] for pair in labels_words] # Get all the available labels 
        count_label1 = len([label for label in labels if label==label1]) # Find the count of all label1's
        count_label2_label1 = 0
        for index in range(len(labels) - 1):
            if labels[index] == label1 and labels[index+1] == label2: # Count the total number of transition to label2 from label1
                count_label2_label1 += 1
        return count_label2_label1/count_label1

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            #return self.simplified(sentence) , self.var_e(sentence)
            return self.simplified(sentence) 
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
    



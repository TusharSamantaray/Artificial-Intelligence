import sys
import pickle
import string
import pandas as pd
import nltk
import math
import operator
from nltk.corpus import stopwords
nltk.download('stopwords')

'''
This is used to remove the common stop words. As common words like 'the', 'or', 'and'..etc.. occurs 
more frequently in any tweet, they hamper the overall prediction calculation because of their frequency.
'''
stopwords_list = list(stopwords.words('english'))
'''
This function cleans each tweet in the training and testing file. 'str.maketrans' removes each 
punctuation available in 'string.punctuation' in each tweet, and then, any extra blank space is 
removed from the end of the tweets using 'rstrip' and then the whole tweet is converted to lowercase.
'''
def clean_tweets(input_file_name):
    tweets = [] # To store all tweeted words available in the training data
    locations = [] # To store all locations available in the training data
    file = open(input_file_name, 'r')

    for tweet in file:
        tweet = tweet.split()
        location = tweet.pop(0)

        # Store the locations in a separate list
        locations.append(location)

        tweet = " ".join(tweet)
        # Remove punctuation, \n and extra space at the end of each tweet and convert to lower case
        tweets.append(tweet.translate(str.maketrans('', '', string.punctuation + '\n')).rstrip().lower())

    return tweets, locations

# This function finds each unique word as twitter_tokens set, and a token_count dictionary that returns count of each 
# twitter_token in the whole training file
def create_twitter_token_set(tweets):
    twitter_tokens = set()
    token_count = {}
    for tweet in tweets:
        for t in tweet.split():
            if t not in stopwords_list:
                twitter_tokens.add(t)
                if t not in token_count.keys():
                    token_count[t] = 1
                else:
                    token_count[t] += 1
    return twitter_tokens, token_count

# This function trains the bayes model using input training file
def train_bayes(input_file_name):
    # Fetch each words and locations
    tweets, locations = clean_tweets(input_file_name)
    unique_locations = set(locations) # Find the unique locations

    # Fetch twitter tokens. Token counts is not required for bayes and is used for dtree classification
    twitter_tokens, token_count = create_twitter_token_set(tweets)

    # Create a dataframe that would store the total count of each word tweeted per unique location in the training dataset. Since all the words 
    # available in the training data might not be tweeted for each city, we initialise a default value of 1/total number of twitter_tokens
    # (unique words) for all location/word combination.
    words_df = pd.DataFrame(1/len(twitter_tokens), index = list(unique_locations), columns = list(twitter_tokens))

    # Store the probability of getting each location
    location_probability = {location: 0 for location in unique_locations}

    # Calculate the count of each word per location
    for tweet, location in zip(tweets, locations):
        tweet = tweet.split()
        location_probability[location] += 1
        for word in tweet:
            if word not in stopwords_list:
                words_df.at[location, word] += 1

    # Normalise location_probability
    total_tweets = len(tweets)
    location_probability = {location: count/total_tweets for location, count in location_probability.items()}

    return words_df, location_probability

# This function trains the dtree model using input training file
def train_dtree(input_file_name, dtree_depth = 12, individual_word_count = 100):
    # Prepare the training data by converting each tweet to a list of feature vectors
    training_data = prepare_data_for_dtree(input_file_name, individual_word_count)

    # Build decision tree
    return dtree_functions.build_tree(training_data, dtree_depth)

# This function converts each tweet into a single integer lists of feature vector, based on the frequency of each word available in a tweet
def prepare_data_for_dtree(input_file_name, individual_word_count):
    # Fetch each words and locations
    tweets, locations = clean_tweets(input_file_name)

    # Fetch twitter tokens
    twitter_tokens, token_count = create_twitter_token_set(tweets)

    # Sort the twitter tokens counts in descending order
    token_count = dict( sorted(token_count.items(), key=operator.itemgetter(1),reverse=True))

    # Select only those twitter tokens which occurs more than value of 'individual_word_count' in the training file. 
    # Considering words which repeat more overall gives us a well populated feature vector that helps in classification.
    relevant_twitter_token_counts = {k: v for (k, v) in token_count.items() if v > individual_word_count}
    global relevant_twitter_tokens # This is made global to use the tokens in Question class
    relevant_twitter_tokens = list(relevant_twitter_token_counts.keys())
    
    # Use this dataframe to form the feature vector of training data
    train_data = pd.DataFrame(0, index = list(range(len(tweets))), columns = relevant_twitter_tokens)

    # Convert each occurance of words to interger format, forming the feature vector
    for i, tweet in enumerate(tweets):
        tweet = tweet.split()
        for word in tweet:
            if word not in stopwords_list and word in relevant_twitter_tokens:
                train_data.at[i, word] += 1

    train_data['actual_tweet_locations'] = locations # Append the actual locations in the training file for each tweet
    # Convert into list format as it is much simpler to partition the data into 'true_words' and 'false_words' in the partition step
    return train_data.values.tolist() 
'''
This class is used to set a column number and column value which acts as the current question, based on which we 
partition the data. For example, if the column number 0 is set to value 1, we partition all values as false_words 
if the previously set column number for each row is less than 1, else they are partitioned as true_words
'''
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    # This function returns true of compares feature in the question with the feaure in the example row
    def is_match(self, row):
        val = row[self.column]
        return val > self.value

    def __repr__(self):
        # It is a helper function to print the question in a readable format.
        return "Is %s %s %s?" % (
            relevant_twitter_tokens[self.column], ">", str(self.value))

# This class is used to store the Leaf node's predictions
class Leaf:
    def __init__(self, word_list):
        self.predictions = dtree_functions.location_count(word_list)

# This class is used to store a question and the branches formed of 'true_words' and 'false_words' based on whether 
# each word is available or not for a location
class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# These are the class of function which contain all the operation of forming, printing a decision tree and predicting 
# locations from a built decision tree
class dtree_functions:
    # This function partitions the given dataset into true or false sets based on the question
    def partition(word_list, question):
        true_words, false_words = [], []
        for row in word_list:
            # For a column value set, partition a word as true_word or false_word if the value 
            # in that row for that word is greater than the value set for the question
            true_words.append(row) if question.is_match(row) else false_words.append(row)
        return true_words, false_words

    # This function returns the counts of each location in the given dataset
    def location_count(word_list):
        counts = {}
        for row in word_list:
            location = row[-1]
            if location not in counts:
                counts[location] = 0
            counts[location] += 1
        return counts

    # This function computes the gini impurity of a given dataset
    def gini_impurity(word_list):
        impurity = 1
        counts = dtree_functions.location_count(word_list)
        for tweet_location in counts:
            p_location = counts[tweet_location]/float(len(word_list))
            impurity -= p_location**2
        return impurity

    # This function computes the information gain of a given set of true set and false set and the current computed uncertainity
    def info_gain(true_words, false_words, current_uncertainity):
        p = float(len(true_words)) / (len(true_words) + len(false_words))
        return current_uncertainity - p * dtree_functions.gini_impurity(true_words) - (1 - p) * dtree_functions.gini_impurity(false_words)

    # This function builds a decision tree
    def build_tree(word_list, depth):
        gain, question = dtree_functions.find_best_split(word_list)
        
        # Stop further branching if gain is 0, or if maximum depth it reached. Set the current training data as the leaf node
        if gain == 0 or depth == 0:
            return Leaf(word_list)

        # Find the true and false words partitions
        true_words, false_words = dtree_functions.partition(word_list, question)

        # Reduce the depth
        depth -= 1

        # Build decision tree for the true branch
        true_branch = dtree_functions.build_tree(true_words, depth)

        # Build decision tree for the false branch
        false_branch = dtree_functions.build_tree(false_words, depth)

        # Return the decision node
        return Decision_Node(question, true_branch, false_branch)

    # This function find the best split of data
    def find_best_split(word_list):
        best_gain = 0
        best_question = None
        current_uncertainity = dtree_functions.gini_impurity(word_list)
        number_of_words = len(word_list[0]) - 1

        for col in range(number_of_words): 
            values = set([row[col] for row in word_list])

            for val in values:
                question = Question(col, val) # Form a current question, 'for instance is word at column 2 of each row greater than or equal to a value 1 ?'

                # Partition the database
                true_words, false_words = dtree_functions.partition(word_list, question)

                # We can skip a split if there is no partition
                if len(true_words) == 0 or len(false_words) == 0:
                    continue
                
                # Compute the information gain of the partitioned words
                gain = dtree_functions.info_gain(true_words, false_words, current_uncertainity) 

                if gain >= best_gain: # Track the question with the best informatio gain value
                    best_gain, best_question = gain, question

        return best_gain, best_question

    # This function print the computed decision tree
    def print_dtree(node, depth = 3):
        # Check if we have reached a Leaf
        if isinstance(node, Leaf):
            print("Predict", node.predictions)
            return

        # Stop is maximum depth is reached
        if depth == 0:
            print(str(node.question))
            return
        
        # Print the question of the current node of the decision tree
        print(str(node.question))
        
        depth -= 1 # Decrease the depth

        # Recursive call over the true branch
        print('===== True Branch =====')
        dtree_functions.print_dtree(node.true_branch, depth)

        # Recursive call over the false branch
        print('===== False Branch =====')
        dtree_functions.print_dtree(node.false_branch, depth)

    # This function is used to predict locations using decision tree
    def location_predict(tweet, dtree):
        if isinstance(dtree, Leaf): 
            return dtree.predictions # Return if current branch is a Leaf

        # Traverse all branches till we find a Leaf
        if dtree.question.is_match(tweet):
            return dtree_functions.location_predict(tweet, dtree.true_branch)
        else:
            return dtree_functions.location_predict(tweet, dtree.false_branch)

    # This function returns the location with maximum predicted probability
    def predicted_location(location_counts):
        total = sum(location_counts.values()) * 1.0
        p = {}
        for location in location_counts.keys():
            p[location] = int(location_counts[location] / total * 100)
        return max(p, key=p.get)

# This function shows the most 5 frequent words for each location
def top_5_words_per_location(words_df):
    frequent_words = pd.DataFrame({n: words_df.T[col].nlargest(5).index.tolist() for n, col in enumerate(words_df.T)}).T
    frequent_words.columns = ['word 1', 'word 2', 'word 3', 'word 4', 'word 5']
    frequent_words.index = list(words_df.index)
    print("The top 5 frequent words for each location are:")
    print(frequent_words)

# This function computes the most probable location of each tweet in test file using bayes classifier.
def compute_bayes(model_input_file, input_file_name, output_file_name):
    estimated_locations = [] # Maintain locations of each test tweet
    actual_locations = [] # Store the actual location in the test tweet. Use it to compute accuracy

    # Load the trained model
    model_to_load = open(model_input_file, "rb")
    words_df = pickle.load(model_to_load)

    input_file = open(input_file_name, 'r')
    #words_df = pd.read_csv(model_input_file, index_col=0) # Load the previously trained model
    
     # Load the previously computed location probabilities
    file_to_read = open("location_probability_file", "rb")
    location_probability = pickle.load(file_to_read)

    # Clean testing data and get each available location
    test_tweets, locations = clean_tweets(input_file_name)
    unique_locations = list(set(locations))

    # Fetch twitter tokens
    twitter_tokens = create_twitter_token_set(test_tweets)

    with open(output_file_name, 'w') as output_file: # Open the output file
        for test_tweet in input_file: # Iterate over each tweet
            original_tweet = test_tweet
            p = [] # Used to maintain a list of computed probabilities of all the cities
            test_tweet = test_tweet.split() # Split the tweet into individual words
            actual_location = test_tweet.pop(0) # Store the actual location in the test tweet. Use it to compute accuracy
            actual_locations.append(actual_location) # Capture the actual location available in test data

            for location in unique_locations: # Compute Naive Bayes probability for each location for each tweet
                location_prob = 1
                for word in test_tweet: # Iterate over each word in a tweet and calculate the P(location | word1, word2... wordn)
                    word = word.translate(str.maketrans('', '', string.punctuation + '\n')).rstrip().lower() # Clean the word
                    if word not in stopwords_list:
                        try:
                            location_prob = location_prob * words_df.loc[location, word] # If [location, word] key is available in dictionary
                        except:
                            location_prob = location_prob * 1/len(twitter_tokens) # If [location, word] key is not available in dicitonary
                location_prob = location_prob * location_probability[location]
                p.append(location_prob)
            
            most_probable_location = unique_locations[p.index(max(p))] # Find the location with highest computed probability
            estimated_locations.append(most_probable_location)

            # Write the estimated locations to the output file
            output_file.write(most_probable_location + ' ' + original_tweet)
    output_file.close() # Close the output file

    # Calculate the accuracy of each estimated location
    calculate_accuracy(actual_locations, estimated_locations)

# This function computes the most probable location of each tweet in test file using dtree classifier
def compute_dtree(model_input_file, input_file_name, output_file_name):
    estimated_locations = [] # Maintain locations of each test tweet
    
    # Load the trained model
    model_to_load = open(model_input_file, "rb")
    dtree = pickle.load(model_to_load)

    input_file = open(input_file_name, 'r')
    # Clean testing data and get each available location. 
    # Actual location stores the actual location in the test tweet. Use it to compute accuracy.
    test_tweets, actual_locations = clean_tweets(input_file_name)

    # Fetch the unique words and the total count from the test file
    test_twitter_tokens, test_token_count = create_twitter_token_set(test_tweets)

    # Create a dataframe to convert each tweet in the test file to numeric feature vector
    test_data = pd.DataFrame(0, index = list(range(len(test_tweets))), columns = list(test_twitter_tokens))

    # Convert the test_twitter tokens to numeric feature vector
    for i, tweet in enumerate(test_tweets):
        tweet = tweet.split()
        for word in tweet:
            if word not in stopwords_list and word in test_twitter_tokens:
                test_data.at[i, word] += 1

    # Convert into list format as it is much simpler to partition the data into 'true_words' and 'false_words' in the partition step
    test_data = test_data.values.tolist()

    # Test each file and write it into the output file
    with open(output_file_name, 'w') as output_file: # Open the output file
        for test_tweet, actual_test_tweet in zip(test_data, input_file):
            current_estimated_location = dtree_functions.predicted_location(dtree_functions.location_predict(test_tweet, dtree)) # Predict location
            estimated_locations.append(current_estimated_location)
            output_file.write(current_estimated_location + ' ' + actual_test_tweet) # Write the prediction of each tweet to output file
    output_file.close() # Close the output file

    # Calculate the accuracy of each estimated location
    calculate_accuracy(actual_locations, estimated_locations)

# This function computes the accurage of the estimated locations
def calculate_accuracy(actual_locations, estimated_locations):
    match_count = 0
    for actual, predicted in zip(actual_locations, estimated_locations):
        if actual == predicted:
            match_count += 1
    print("Location prediction accuracy: " + str(match_count/len(actual_locations)))

# Main function
if __name__ == "__main__":
    '''
    if(len(sys.argv) != 5):
        raise(Exception("Error: Please provide valid inputs in the following format: geolocate.py train/test bayes/dtree training/test_file_name output_file_name"))

    # Get parameterized inputs
    mode = sys.argv[1]
    if mode == 'train':
        (classifier, input_file_name, output_file_name) = sys.argv[2:]
    elif mode == 'test':
        (model_input_file, input_file_name, output_file_name) = sys.argv[2:]
    else:
        raise(Exception("Error: Please provide valid mode: 'train' or 'test'"))
    '''
    # For testing bayes
    #(mode, classifier, input_file_name, output_file_name) = 'train', 'bayes', 'tweets.train.clean.txt', 'bayes_training_file'
    #(mode, model_input_file, input_file_name, output_file_name) = 'test', 'bayes_training_file', 'tweets.test1.clean.txt', 'bayes_output.txt'
    #python3 geolocate.py train bayes tweets.train.clean.txt bayes_training_file
    #python3 geolocate.py test bayes_training_file tweets.test1.clean.txt bayes_output.txt

    # For testing dtree
    #(mode, classifier, input_file_name, output_file_name) = 'train', 'dtree', 'tweets.train.clean.txt', 'dtree_training_file'
    (mode, model_input_file, input_file_name, output_file_name) = 'test', 'dtree_training_file', 'tweets.test1.clean.txt', 'dtree_output.txt'
    #python3 geolocate.py train dtree tweets.train.clean.txt dtree_training_file
    #python3 geolocate.py test dtree_training_file tweets.test1.clean.txt dtree_output.txt

    if mode == 'train':
        if classifier == 'bayes':
            print("---- Naive Bayes classifier training started ----")
            trained_model, location_probability = train_bayes(input_file_name)
            print("---- Naive Bayes classifier training completed ----")
            # Display 5 most frequent words in each location
            top_5_words_per_location(trained_model)
        elif classifier == 'dtree':
            print("---- Decision tree classifier training started ----")
            # 2nd parameters is the depth of decision tree and 3rd parameters is the individual word count
            # Words appearning more than individual word counts are only considered to construct feature vector
            trained_model = train_dtree(input_file_name, 50, 500)
            print("---- Decision tree classifier training completed ----")
            dtree_functions.print_dtree(trained_model)  
        else:
            raise(Exception("Error: Please provide valid classifier: 'bayes' or 'dtree'"))

        # Save the trained model classifier type
        c = {output_file_name: classifier}
        save_model_type = open('model_type', 'wb')
        pickle.dump(c, save_model_type)
        save_model_type.close()

        # Save the trained model
        try:
            trained_model_file = open(output_file_name, 'wb')
            pickle.dump(trained_model, trained_model_file)
            trained_model_file.close()
            print(classifier + " classifer saved as " + output_file_name)
        except:
            raise(Exception("Error: Failed to save the trained model to file"))

        # Save the location probability for bayes classifier in a file to use while testing
        if classifier == 'bayes':
            try:
                location_probability_file = open('location_probability_file', 'wb')
                pickle.dump(location_probability, location_probability_file)
                location_probability_file.close()
            except:
                raise(Exception("Error: Failed to save computed location_probabilites to file"))

    elif mode == 'test':
        # Save the trained model classifier type
        file_to_load = open('model_type', 'rb')
        words_df = pickle.load(file_to_load)
        classifier = words_df[model_input_file]

        # Test the classifier
        if classifier == 'bayes':
            compute_bayes(model_input_file, input_file_name, output_file_name)
        elif classifier == 'dtree':
            compute_dtree(model_input_file, input_file_name, output_file_name)
        else:
            raise(Exception("Error: Please provide a valid classifier: 'bayes' or 'dtree'"))
    else:
        raise(Exception("Error: Please provide valid mode: 'train' or 'test'"))

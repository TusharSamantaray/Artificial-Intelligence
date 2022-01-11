## Naïve Bayes Classifier

#### Approach

- From the given 'tweet.train.clean.txt', a 'words_df' dataframe was constructed that contains the frequency of all the words that were found in each tweet in the training data file. Similarly, a dictionary called 'location_probability' was populated with probability of finding each location in the training data file. 
- To predict the location of a tweet, the posterior probability was calculated as Max(P(location) * P(each word in the tweet | location)).

#### Functions used - 

###### For training - 

- train_bayes(input_file_name) - This function takes an input training file name and returns a words_df dataframe which consists of count of each word per available location in the training file and a dictionary of probability of getting each location.
- clean_tweets(input_file_name) - This function takes an input training file name and returns a list of tweets by removing punctuations and stop words(commonly occurring words that wouldn't help much in predicting tweet locations).
- create_twitter_token_set(tweets) - This function returns each individual word available in the tweets called as twitter_tokens and the total number of times each word appeared. The total_counts is only used for decision tree classifier.
- top_5_words_per_location(words_df) - This function takes the words_df dataframe returned by train_bayes method as input and displays the top 5 words per location.

###### For testing - 

- compute_bayes(model_input_file, input_file_name, output_file_name) - This function take the file name of the previously saved bayes model, the test input file name and a file name to save the output of the location predictions using bayes followed by the original location and tweet. 
- calculate_accuracy(actual_locations, estimated_locations) - This function takes a list of the actual locations of each tweets and a list of estimated locations and prints the accuracy of the predictions.

My initial approach was to form the word_df and compute the location_probabilities and predict the locations. This gave me an accuracy of 0.582. However, removing the stop words by using nltk package increased the accuracy to 0.608. I also tried to split combined words like 'mynameisTushar' to 'my name is Tushar' using the open source 'wordninja' pip package but that reduced the accuracy to 0.508 and hence I decided to not use it. 

[This article](https://www.mygreatlearning.com/blog/pos-tagging/) on POS tagging from previous assignment again proved to be insightful in approaching this problem.

## Decision Tree Classifier

#### Approach - 

- To build a decision tree classifier, the first step is to transform the input training data into a format suitable to train a decision tree. This post proved helpful to provide a correct way to transform the data into a set of feature vectors of equal lengths (this is important as each tweet is of different length). 
- Since the formed feature vector of words from 32,000 tweet were more than 47,000, I decided to use only use only such features that were more frequent in the training file to speed up the decision tree creation process.
- To predict, each tweet is classified over the branches of the built decision tree till we reach a leaf prediction node. Once we get to a leaf node, the node with the highest probability is retuned as the result.

#### Functions used - 

###### For training - 

- train_dtree(input_file_name, dtree_depth, individual_word_count) - This function takes a input training file name, the depth of the decision tree to be build and the word count above which the features will be considered, for instance all words that occurred less than the 'individual_word_count' (say 10 for example), then such words won't be considered as a feature vector. Since the feature vector for the provided training file was above 47,000, this helped reduced computation time significantly, and returns a built decision tree.
- prepare_data_for_dtree(input_file_name, individual_word_count) - This function returns a feature vector of the words which are more frequent than individual_word_count value.
- clean_tweets(input_file_name) and create_twitter_token_set(tweets) functions for same purpose as in Naïve Bayes classifier.
- print_dtree(node, spacing, depth) - This function is used to print the build decision tree. This takes the current node, starting from root node, as input, and the depth till will the decision tree is print.
- class dtree_functions - This call contains all required functions to build, display and compute a decision tree.
  - build_tree(word_list, depth) - This function takes the build feature vector (named word_list) as input and the depth of the decision tree to be build and returns a decision tree of provided depth.
  - find_best_split(word_list) - This function finds the best decision split of data based on a 'question'. For instance, a question would be something like 'is feature number n has a value more than v?' and each tweet in the training file is split on this to either true or false words depending on if they suffice the above question or not. 
  - gini_impurity(word_list) - This function computes uncertainty, that is the measure of how often a randomly chosen feature's location from the dataset is predicted incorrectly if that was randomly predicted according to the distribution of features in the subset. [Source](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) 
  - partition(word_list, question) - This function partitions the word_list based on the current random question and returns two list of true_words and false_words according to the question.
  - info_gain(true_words, false_words, current_uncertainity) - This function computes the entropy of the formed true_words and false_words sets. The split that provides more information gain is more a more favorable one in making decision. [Source](https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain)
  - location_count(word_list) - This function computes the number of available tweets per location in the current split of data. This is used to compute gini_impurity.
- class Question - 
  - This class hold a column and a value, where the column is a single feature vector (that is a word in the formed feature vector), and a value from all possible values of that feature vector, hence forming the question. For instance 'is cat appearing more than val 2 in a tweet'. Based on this question, each tweet in the training file gets partitioned in a recursive manner to form the decision tree. 
  - The 'is_match' function is used to check if the values of each row for a question column is more than the question value. Returns true or false based on the result.
  - The repr function is used to reformat the question in a readable format. This is helpful to print the built decision tree.
- class Leaf - This class hold the leaf of a branch, which is the final set of predictions of a built branch.
- class Decision_Node - This class is used to define a question and split formed by that question forming the true_branch and false_branch based on the question.

###### For testing -

- compute_dtree(model_input_file, input_file_name, output_file_name) - This function take the file name of the previously saved decision tree model, the test input file name and a file name to save the output of the location predictions using  decision tree followed by the original location and tweet. 
- location_predict(test_tweet, dtree) - This function takes a tweet to test and a decision tree and returns the Leaf of the branch that best predicts the location of the tweet. A Leaf might have more than one predicted location.
- predicted_location(location_predict()) - This function returns the location with highest probability of all the locations available in a leaf returned by location_predict function.

The maximum accuracy I noted for the decision tree was 0.214, though the result is not always consistent for the same test file over the same trained model. The accuracy varied mostly between 0.182 to 0.204. I tried to increase depth of the decision tree till 50 and train over words that appear more than 1 in the entire training file, but I didn't noticed any significant improvement over the accuracy. Splitting compounded words like in the Naïve Bayes classifier also did not had any significant improvement or decrease in the accuracy and it mostly stayed between 0.182 and 0.204.

[This article](https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea) and this  [YouTube video](https://www.youtube.com/watch?v=LDRbO9a6XPU), and the [implementation](https://github.com/random-forests/tutorials/blob/master/decision_tree.py) of the example in the video helped me most to solve this problem and this [stack overflow post](https://stackoverflow.com/questions/48090757/text-classification-using-decision-trees-in-python) was helpful to give me an intuition to build a feature vector from the training data. I also tried to implement tf-idf(term frequency - inverse document frequency) from the following blogs: [part1](https://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/) and [part2](https://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/), which would give  the words most important to each location, instead of just counts of words per location. However, I couldn't figure out how to compute this in a reasonable amount of time (30 mins was also not enough) over 32,000 tweets and over 47,000 words and had to drop it.

To save and retrieve the trained models, location probability and model_type, I used pickle library.




#### Approach

The goal is to decrypt the encoded document by replacing every letter and spaces with the appropriate letters and rearranging the sequence of letters and space of length 4 among themselves. 

The approach used to decrypt the document is as follows,
1.	current_replacement_guess : This generates English alphabets in lower case and shuffles it randomly. This list is used as a initial cipher to start with.
2.	Since the document is broken into chucks of length four, initial arrangement is considered as [0, 1, 2,3]
3.	create_alphabet_transition_dict(): This function takes the corpus file as a input to build the transition between two letters in the document. Transition between two letters is stored in the dictionary with the value as number of times the transition appeared in the corpus.
4.	generate_new_replacement_guess (): This function takes the current cipher and generates the new cipher by just swap two letter, where it’s position is randomly generated.
5.	rearrange_encrypted_document (): This function takes encrypted string and the random arrangement as input and splits the encrypted string equally with the chunks of length 4. The order in every chunk is arranged based on the newly generated arrangement.
6.	generate_new_rearrangement_guess(): Generates new arrangement.
7.	get_decrypted_document_score(): Cipher score is generated for every new cipher. This function takes encrypted file, cipher and alphabet transition dictionary as input to calculate the cipher score. The document is decrypted using the function, decrypt_document() and new alphabet transition is constructed. Log value of every key in new alphabet transition is calculated and added together as cipher score.
8.	Cipher score is calculated for every cipher generated and decrypted. Best score so far is stored. The process of decryption is performed with the best cipher score.


We tried to understand the application of MCMC and Metropolis-Hastings algorithm using various online resources. [The article](https://mlwhiz.com/blog/2015/08/21/mcmc_algorithm_cryptography/), Behold the power of MCMC provided us good insight and understanding on breaking the code and decrypt the document. We did not achieve 100% decryption of the document. We also tried to normalize the transition data, hoping to achieve better result.
# main.py

## Introduction
This code in python tests function word context independence by training an SVM model to classify them in different text types. It randomly selects texts, creates a dictionary and converts the texts into bag-of-words vectors. It trains two SVM models, one using word counts and the other using the training data, and evaluates their performance on both training and test datasets. The results are stored in a dictionary for each iteration.

## Installation
For dependencies, use:

    $ pip install -r requirements.txt

The requirements file was generated by the pipreqs library. Used Python version 3.9.12, run in Anaconda. No other installation are needed to run the script.

## Input
The code assumes text files as input with the following structure:

	Data/text_type1/textfile1.txt
			   /textfile2.txt
                     /...

	    /text_type2/textfile1.txt
			   /textfile2.txt
			   /...
	    /...


Text files content is supposed to contain one function word per line, and lines separated by \n, e.g.:

    the
    although
    while
    the
    ...


## Output
The output is a dictionary in json format containing results for evaluation metrics ("accuracy", "precision", "recall", "f1") for training ("train") and testing ("test") data represented as a bag-of-words vector for every examined length (e.g., "100") for every text types pair (e.g., "plos_vs_novels"). It also contains accuracy for training and testing data for a counts vectorization ("accuracy_counts"), as well as information about the number of texts for a given length for every text types pair ("lesser_num_of_texts"). 

## Settings
The code allows changing several parametres, namely:

- **BOW_TYPE**: Set "BINAR" or "FREQS" for an according bag-of-words representation (default: "BINAR").
- **LEMMATIZE**: Set True or False for enable/disable lemmatization (default: True).
- **HOW_MANY_QUALIFYING_FW**: Set a number of function words that will form global dictionary (default: 50).
- **KERNEL_TYPE**: Set the kernel type of the SVM model (default: "linear").
- **RESAMPLE**: Set the number of iterations (default: 100).
- **MIN_FILE_COUNT**: Set the minimum number of files necessary to start the loop for a given pair (default: 5).
- **TYPES_OF_FW_SELECTION**: Give a list of all possible methods of function words selection (default: ["MOST_FREQUENT_FROM_EACH_TEXT", "MOST_FREQUENT", "RANDOM"]). The method currently used is further specified in WORD_PICK_TYPE (default: TYPES_OF_FW_SELECTION[0]).
- **LENGTHS**: Give a list of lengths of interest the text will be shortened to (default: [50, 100, 300, 500, 1000]).
- **TEXTS_TYPES**: Give a list of text types (default: ["plos", "novels", "blogs", "reviews", "poems"]).


# analysis.R
This script in R processes JSON output from the Python Loop script and exports the required tables and charts.

# fw_extraction.py
This script in python extracts the function words from the raw texts based on agreement of spacy and nltk pos-taggers. It also filters out words on a stop-list, all non-alphabetic characters, as well as individual letters (with exception of a/i).
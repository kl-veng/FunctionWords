# SETTINGS
BOW_TYPE = "BINAR"              # choose "BINAR" or "FREQS"
HOW_MANY_QUALIFYING_FW = 50     # how many function words will form global dictionary
KERNEL_TYPE = "linear"          # "linear" or "rbf"
RESAMPLE = 100                  # how many times the whole cycle is repeated
MIN_FILE_COUNT = 5

TYPES_OF_FW_SELECTION = ["MOST_FREQUENT_FROM_EACH_TEXT", "MOST_FREQUENT", "RANDOM"]
LENGTHS = [50, 100, 300, 500, 1000]
TEXTS_TYPES = ["plos", "novels", "blogs", "reviews", "poems"]

WORD_PICK_TYPE = TYPES_OF_FW_SELECTION[0]

import glob
import ntpath
import re
import itertools
import tqdm
import json
import numpy as np
import random

from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import svm

# Create X and y lists from a dictionary
def get_x_y(dict_bow, y_value, X, y):
    for bow in dict_bow:
        data = dict_bow[bow]
        X.append(data)
        y.append(y_value)
    return X,y

# Gets frequencies of all tokens
def get_tokens_freqs(tokens):
    freqs_dict = {}
    for token in tokens:
        if token not in freqs_dict: 
            freqs_dict.update({token : 1 })
        else: 
            freqs_dict[token] += 1
    return freqs_dict

# Loads a text file
def load_file(file_name):
    with open(file_name, encoding="utf-8") as f:
        content = f.read()
    return content

# Removes empty tokens
def remove_empty_words(tokens):
    tokens = list(filter(None, tokens))  
    return tokens  

# Splits text file by new lines
def split_by_new_line(text):
    tokens = re.split("\\n", text)
    tokens = remove_empty_words(tokens)
    return tokens

# Gets list of files in a directory
def get_files_list_from_folder(way_to_folder):
    return glob.glob(way_to_folder)

# Gets all text files from a directory
def get_files_from_folder(folder):
    result = []
    for file in get_files_list_from_folder(folder + "\\*.txt"):
        my_file = load_file(file)
        file_name = ntpath.basename(file)
        result.append({"name": file_name, "data": my_file})
    return result

# preprocesses training texts (including global dictionary updating)
def preprocess_train_texts(loaded_texts, global_dict, qualifying_words):
    dict_tokens = {}
    dict_bow = {}

    for text in loaded_texts:
        tokens = (text["data"])
        data = []
        
        for token in tokens:
            if token in qualifying_words:
                data.append(token)      
      
        global_dict.update(data)
        dict_tokens[text["name"]] = data
        dict_bow[text["name"]] = []

    return dict_bow, dict_tokens

# preprocesses testing texts (without global dictionary updating)
def preprocess_test_texts(loaded_texts, qualifying_words):
    dict_tokens = {}
    dict_bow = {}

    for text in loaded_texts:
        tokens = (text["data"])
        data = []
        
        for token in tokens:
            if token in qualifying_words:
                data.append(token)      
      
        dict_tokens[text["name"]] = data
        dict_bow[text["name"]] = []

    return dict_bow, dict_tokens

# Creates Bag-of-Words representation
def make_bow(loaded_texts, dict_tokens, dict_bow, global_dict, bow_type):
    for word in global_dict:
        for text in loaded_texts:
            text_name = text["name"]
            tokens = dict_tokens[text_name]
            tokens_freqs = get_tokens_freqs(tokens)
            is_in_text = word in tokens_freqs
            if bow_type == "FREQS":
                dict_bow[text_name].append(tokens_freqs[word] if is_in_text else 0)
            elif bow_type == "BINAR":
                dict_bow[text_name].append(1 if is_in_text else 0)
    return dict_bow  

# Provides evaluation metrics to measure the model's performance
def get_evaluation_metrics(y_true, y_pred):
    precision   = average_precision_score(y_true, y_pred)
    recall      = recall_score(y_true, y_pred)
    accuracy    = accuracy_score(y_true, y_pred)
    f1          = f1_score(y_true, y_pred)
    return precision, recall, accuracy, f1

# Splits texts into a training and testing subset in a given ratio
def split_train_test_texts(loaded_texts, ratio):
    num_of_texts = len(loaded_texts)
    where_to_split = int(num_of_texts*ratio)
    randomized_texts = random.sample(loaded_texts, len(loaded_texts))

    list_train = randomized_texts[:where_to_split]
    list_test  = randomized_texts[where_to_split:]

    return list_train, list_test

# Samples given number of words from every given text that is longer than the given number
def select_longer_than_minimum(loaded_texts, num_of_fw):
    texts = []
    for i in range(len(loaded_texts)):
        name = loaded_texts[i]["name"]
        tokens = split_by_new_line(loaded_texts[i]["data"])
        if len(tokens) >= num_of_fw:
            data = random.sample(tokens, num_of_fw)
            texts.append({"name": name, "data": data})
    return texts        

# Gets a list of frequencies
def calculate_frequencies(texts, list_of_frequencies):
    for i in range(len(texts)):
        fw = texts[i]["data"]
        num_tokens = len(fw)
        fw_freqs = get_tokens_freqs(fw)
        for token in fw_freqs:
            fw_freqs[token] = (fw_freqs[token]/num_tokens)*100                 
        ordered_fw_freqs = sorted(fw_freqs.items(), key=lambda val: val[1], reverse=True)
        list_of_frequencies.append(ordered_fw_freqs)
    return list_of_frequencies

# Saves the result dictionary to a given path
def save_dictionare(file_name, dictionare):
    with open(file_name, 'w') as fp:
        json.dump(dictionare, fp)

########################################
results = {}
print(f"** {WORD_PICK_TYPE}   ***********************************")

for pair in tqdm.tqdm(itertools.combinations(TEXTS_TYPES, 2)):
    length_results = {}
    FIRST_CATEGORY, SECOND_CATEGORY = pair
    print(f"... working on {FIRST_CATEGORY} vs. {SECOND_CATEGORY}...")

    for length in tqdm.tqdm(LENGTHS, "\tLength"):
        precision_list_train = []
        recall_list_train = []
        accuracy_list_train = []
        f1_list_train = []
        accuracy_linear_counts_train = []

        precision_list_test = []
        recall_list_test = []
        accuracy_list_test = []
        f1_list_test = []
        accuracy_linear_counts_test = []

        print("Loading text files...")      # 1. Load data in plain text format, ensure the same number of files for each group
        loaded_texts_A_all = get_files_from_folder(f"Data/fw_{FIRST_CATEGORY}")
        loaded_texts_B_all = get_files_from_folder(f"Data/fw_{SECOND_CATEGORY}") 

        texts_A = select_longer_than_minimum(loaded_texts_A_all, length)
        texts_B = select_longer_than_minimum(loaded_texts_B_all, length)

        num_of_texts_A = len(texts_A) 
        num_of_texts_B = len(texts_B)
        lesser_num_of_texts = min(num_of_texts_A, num_of_texts_B)
        print(f"number of texts: {lesser_num_of_texts}")
        
        if lesser_num_of_texts > MIN_FILE_COUNT:
            for k in range(RESAMPLE):
                print(f"Resample: {k}")

                loaded_texts_A = random.sample(texts_A, lesser_num_of_texts)
                loaded_texts_B = random.sample(texts_B, lesser_num_of_texts)

                print("Splitting training and testing data...")
                A_train, A_test = split_train_test_texts(loaded_texts_A, 4/5)
                B_train, B_test = split_train_test_texts(loaded_texts_B, 4/5)

                # first option: pick 50 most frequent FW from the whole corpus
                if WORD_PICK_TYPE == "MOST_FREQUENT":
                    freqs = []
                    freqs = calculate_frequencies(A_train, freqs)
                    freqs = calculate_frequencies(B_train, freqs)

                    print("Calculating average of fw percentige...")
                    dict_freqs = {}
                    for i in range(len(freqs)):
                        for key in freqs[i]:
                            if key[0] not in dict_freqs:
                                dict_freqs[key[0]] = []

                    for key in dict_freqs:
                        for i in range(len(freqs)):
                            for j in range(len(freqs[i])):
                                if key in freqs[i][j]:
                                    dict_freqs[key].append(freqs[i][j][1])

                    for key in dict_freqs:
                        dict_freqs[key] = sum((dict_freqs[key]))
                        dict_freqs[key] = dict_freqs[key]/len(freqs)

                    ordered_dict = sorted(dict_freqs.items(), key=lambda val: val[1], reverse=True)
                    ordered_dict[0:HOW_MANY_QUALIFYING_FW]

                    qualifying_words = ordered_dict[0:HOW_MANY_QUALIFYING_FW]
                    qualifying_words = [x[0] for x in qualifying_words]

                # second option: pick n most frequent FW from each text
                if WORD_PICK_TYPE == "MOST_FREQUENT_FROM_EACH_TEXT":
                    qualifying_words_list = []

                    for i in range(len(A_train)):
                        fw = A_train[i]["data"]
                        fw_freqs = get_tokens_freqs(fw)
                        ordered_fw_freqs = sorted(fw_freqs.items(), key=lambda val: val[1], reverse=True)
                        ordered_fw = [x[0] for x in ordered_fw_freqs]
                        qualifying_words_list.extend(ordered_fw[0:HOW_MANY_QUALIFYING_FW])

                    for i in range(len(B_train)):
                        fw = B_train[i]["data"]
                        fw_freqs = get_tokens_freqs(fw)
                        ordered_fw_freqs = sorted(fw_freqs.items(), key=lambda val: val[1], reverse=True)
                        ordered_fw = [x[0] for x in ordered_fw_freqs]
                        qualifying_words_list.extend(ordered_fw[0:HOW_MANY_QUALIFYING_FW])
                    
                    qualifying_words = set(qualifying_words_list)

                # third option: pick randomly n FW
                if WORD_PICK_TYPE == "RANDOM":
                    all_fw = []
                    for i in range(len(A_train)):
                        fw = A_train[i]["data"]
                        all_fw.extend(fw)

                    for i in range(len(B_train)):
                        fw = B_train[i]["data"]
                        all_fw.extend(fw)
                    
                    qualifying_words = random.sample(set(all_fw), HOW_MANY_QUALIFYING_FW)

                print(qualifying_words)

                ##############################################################################
                global_dict = set()

                print("Preprocessing of training data...")       
                dict_bow_A_train, dict_stems_A_train = preprocess_train_texts(A_train, global_dict, qualifying_words)
                dict_bow_B_train, dict_stems_B_train = preprocess_train_texts(B_train, global_dict, qualifying_words)
                print(len(global_dict))

                print("Preprocessing of testing data...")
                dict_bow_A_test,  dict_stems_A_test  = preprocess_test_texts(A_test, qualifying_words)
                dict_bow_B_test,  dict_stems_B_test  = preprocess_test_texts(B_test, qualifying_words)

                print("Creating BoWs...")
                dict_bow_A_train = make_bow(A_train, dict_stems_A_train, dict_bow_A_train, global_dict, BOW_TYPE)
                dict_bow_A_test  = make_bow(A_test,  dict_stems_A_test,  dict_bow_A_test,  global_dict, BOW_TYPE)
                dict_bow_B_train = make_bow(B_train, dict_stems_B_train, dict_bow_B_train, global_dict, BOW_TYPE)
                dict_bow_B_test  = make_bow(B_test,  dict_stems_B_test,  dict_bow_B_test,  global_dict, BOW_TYPE)

                ################# evaluation of training data:
                X_train, y_train = [], []
                X_train, y_train = get_x_y(dict_bow_A_train, 1, X_train, y_train)
                X_train, y_train = get_x_y(dict_bow_B_train, 0, X_train, y_train)
                
                print("Training your model...")
                model  = svm.SVC(kernel = KERNEL_TYPE)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_train)
                average_precision, recall, accuracy, f1 = get_evaluation_metrics(y_train, y_pred)

                precision_list_train.append(average_precision)
                recall_list_train.append(recall)
                accuracy_list_train.append(accuracy)
                f1_list_train.append(f1)

                print("Training model on counts...")
                counts_x_train = np.array(X_train).sum(axis=1).reshape(-1, 1)                
                counts_model = svm.SVC(kernel = "linear")
                counts_model.fit(counts_x_train, y_train)
                accuracy_linear_counts_train.append(
                    get_evaluation_metrics(y_train, counts_model.predict(counts_x_train))[2]
                )
                
                ################### evaluation of testing data:
                X_test, y_test =  [], []
                X_test, y_test = get_x_y(dict_bow_A_test, 1, X_test, y_test)
                X_test, y_test = get_x_y(dict_bow_B_test, 0, X_test, y_test)

                print("Making predictions...")
                y_pred = model.predict(X_test)
                average_precision, recall, accuracy, f1 = get_evaluation_metrics(y_test, y_pred)
                
                precision_list_test.append(average_precision)
                recall_list_test.append(recall)
                accuracy_list_test.append(accuracy)
                f1_list_test.append(f1)

                counts_x_test  = np.array(X_test).sum(axis=1).reshape(-1, 1)
                accuracy_linear_counts_test.append(
                    get_evaluation_metrics(y_test, counts_model.predict(counts_x_test))[2]
                )

        length_results[length] = {
            "lesser_num_of_texts" : lesser_num_of_texts,
            "train": {
                "accuracy": accuracy_list_train, 
                "precision": precision_list_train, 
                "recall": recall_list_train, 
                "f1": f1_list_train, 
                "accuracy_counts" : accuracy_linear_counts_train,
            }, 
            "test": {
                "accuracy": accuracy_list_test, 
                "precision": precision_list_test, 
                "recall": recall_list_test, 
                "f1": f1_list_test,
                "accuracy_counts" : accuracy_linear_counts_test
            }
        }

    results[f"{FIRST_CATEGORY}_vs_{SECOND_CATEGORY}"] = length_results

save_dictionare(f"results_{KERNEL_TYPE}_{WORD_PICK_TYPE}.json", results)
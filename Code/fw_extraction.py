# get function words from text files with double check by spacy and nltk, clean them

# SETTINGS
LEMMATIZE = True # set True or False

import glob
import ntpath
import re
import spacy
import nltk

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 20000000

def load_file(file_name):
    with open(file_name, encoding="utf-8") as f:
        content = f.read()
    return(content)

def get_files_list_from_folder(way_to_folder):
    return glob.glob(way_to_folder)
   
def get_files_from_folder(folder):
    result = []
    for file in get_files_list_from_folder(folder + "\\*.txt"):
        my_file = load_file(file)
        file_name = ntpath.basename(file)
        result.append({"name": file_name, "data": my_file})
    return result

########################################
print("Loading text files...")

loaded_texts = get_files_from_folder("Data/reviews") # select text type to be processed: reviews blogs poems plos novels
stoplist = ["amid", "ugh", "vhat", "thatthat", "overwhat", "allthough", "thinkingthat", "blackhat", "tablehat", "timewhat", "womanwhat", "whatthat", "wordwhat", "thhat", "wishthat", "wasthat", "tthat"]

for i in range(len(loaded_texts)):
    print(i)
    text_fw = []
    data = str(loaded_texts[i]["data"])
    nlp_data = nlp(data)

    WANTED_POS_SPACY = ["ADP", "CONJ", "CCONJ", "SCONJ", "DET"]
    for j in range(len(nlp_data)):        
        if nlp_data[j].pos_ in WANTED_POS_SPACY:
            if LEMMATIZE == True:
                token = str(nlp_data[j].lemma_)
            else: token = str(nlp_data[j])
            text_fw.append(token.lower())

    fw = []
    for j in range(len(text_fw)):
        if text_fw[j] == "vs." or text_fw[j] == "v.":
            text_fw[j] = "vs"
        if re.search(".*[^A-Za-z]+.*", text_fw[j]) == None and re.search("[A-Za-z]{2,}|[a,i]", text_fw[j]) != None:
            fw.append(text_fw[j])
   
    WANTED_POS_NLTK = ["CC", "DT", "IN", "PDT", "RP", "TO", "WDT"]
    pre_final_list = list(filter(lambda word: nltk.pos_tag([word])[0][1] in WANTED_POS_NLTK, fw))
    final_list = list(filter(lambda word: word not in stoplist, pre_final_list))    
    if len(final_list) > 30:
        with open("Data/fw_reviews/fw_reviews_" + str(i) + ".txt", mode="x", encoding = "utf-8") as my_file:
            for word in final_list:
                my_file.write(word + "\n")
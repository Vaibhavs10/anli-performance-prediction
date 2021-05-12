import numpy as np
from csv import reader
from collections import defaultdict 

def parse_and_return_rows(file_path):
    with open(file_path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
    return list_of_rows

def return_len_and_vocabulary(list_of_rows):
    #clean the corpus.
    sentences = []
    vocab = []
    for sent in list_of_rows:
        new_sent = sent[1] + " " + sent[2] + " " + sent[3] + " " + sent[4]
        x = new_sent.split(" ")
        sentence = [w.lower() for w in x if w.isalpha() ]
        sentences.append(sentence)
        for word in sentence:
            if word not in vocab:
                vocab.append(word)
    return vocab, len(vocab)

def create_token_index(vocab):
    index_word = {}
    i = 0
    for word in vocab:
        index_word[word] = i 
        i += 1
    return index_word

def return_count_vector(sent, index_word, len_vector):
    count_dict = defaultdict(int)
    vec = np.zeros(len_vector)
    for item in sent:
        count_dict[item] += 1
        for key,item in count_dict.items():
            vec[index_word[key]] = item
    return vec

def preprocess_sentence(sent):
    return [w.lower() for w in sent.split(" ") if w.isalpha()]
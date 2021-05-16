import numpy as np
from csv import reader
from collections import defaultdict 

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_and_return_rows(file_path):
    with open(file_path, 'r', encoding='utf-8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
    return list_of_rows

def return_len_and_vocabulary(list_of_rows):
    #clean the corpus.
    vocab = set()
    for sent in list_of_rows:
        new_sent = sent[1] + " " + sent[2] + " " + sent[3] + " " + sent[4]
        sentence = preprocess_sentence(new_sent)
        vocab |= {word for word in sentence}
    return list(vocab), len(vocab)

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
            try:
                vec[index_word[key]] = item
            except:
                pass
    return vec

def sparse_incidence_vector(sentence, word_to_index):
    """
    Given a string and a dictionary that maps words to their index in the vocabulary, 
    returns a set containing all the indexes of the words that the sentence contains.
    """
    return { word_to_index[w] for w in preprocess_sentence(sentence) if w in word_to_index }

def preprocess_sentence(sent):
    return [w.lower() for w in sent.split(" ") if w.isalpha()]

def fit_tfidf_vectorizer(corpus):
    vectorizer = TfidfVectorizer()
    vect = vectorizer.fit(corpus)
    return vect

def transform_tfidf_vectorizer(vect, sent):
    return vect.transform(sent)

def create_tfidf_corpus(list_of_rows):
    corpus = []
    for row in list_of_rows:
        sent = row[1] + " " + row[2] + " "  + row[3] + " "  + row[4]
        corpus.append(sent)
    return corpus

def return_tfidf_row_lists(list_of_rows):
    obs1 = []
    obs2 = []
    hyp1 = []
    hyp2 = []

    for row in list_of_rows:
        obs1.append(row[1])
        obs2.append(row[2])
        hyp1.append(row[3])
        hyp2.append(row[4])

    return obs1, obs2, hyp1, hyp2

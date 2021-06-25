import nltk
import data.data_loader as data_loader

def load_data():
    rows = data_loader.parse_and_return_rows("data/processed_data/train.csv")
    documents = [(nltk.tokenize.word_tokenize(row[1].lower()), 
    nltk.tokenize.word_tokenize(row[2].lower()), 
    nltk.tokenize.word_tokenize(row[3].lower()), 
    nltk.tokenize.word_tokenize(row[4].lower()),
    row[5]) for row in rows]

    corpus = [[doc[0], doc[1], doc[2], doc[3]] for doc in documents]
    corpus = [item for sublist in corpus for item in sublist]

    return documents, corpus

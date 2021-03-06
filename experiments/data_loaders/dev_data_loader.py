import nltk
import data.data_loader as data_loader

def load_data():
    rows = data_loader.parse_and_return_rows("data/processed_data/dev.csv")

    # tokenize texts
    documents = [(nltk.tokenize.word_tokenize(row[1].lower()), 
    nltk.tokenize.word_tokenize(row[2].lower()), 
    nltk.tokenize.word_tokenize(row[3].lower()), 
    nltk.tokenize.word_tokenize(row[4].lower()),
    row[5]) for row in rows]

    corpus = [row[0] + row[1] + row[2] + row[3] for row in documents]
        
    return documents, corpus


def load_data_raw():
    rows = data_loader.parse_and_return_rows("data/processed_data/dev.csv")

    documents = [(row[1] + " " + row[2], row[3], row[4], row[5]) for row in rows]
        
    return documents

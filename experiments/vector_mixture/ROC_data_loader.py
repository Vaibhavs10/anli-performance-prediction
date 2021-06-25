import nltk
import data.data_loader as data_loader

def load_data():
    rows = data_loader.parse_and_return_rows("data/ROCStories/ROCStories_winter2017 - ROCStories_winter2017.csv")
    rows = rows[1:] # remove row that contains the header

    documents = [(nltk.tokenize.word_tokenize(row[2].lower()), 
    nltk.tokenize.word_tokenize(row[3].lower()), 
    nltk.tokenize.word_tokenize(row[4].lower()), 
    nltk.tokenize.word_tokenize(row[5].lower()), 
    nltk.tokenize.word_tokenize(row[6].lower())) for row in rows]

    corpus = [[doc[0], doc[1], doc[2], doc[3], doc[4]] for doc in documents]
    corpus = [item for sublist in corpus for item in sublist]

    return documents, corpus
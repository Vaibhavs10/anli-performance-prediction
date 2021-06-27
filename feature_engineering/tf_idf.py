import gensim 

def train_tf_idf_model(texts):
    dictionary = gensim.corpora.Dictionary(texts)
    tfidf_model = gensim.models.TfidfModel([dictionary.doc2bow(text) for text in texts])
    return tfidf_model, dictionary

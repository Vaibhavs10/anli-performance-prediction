import numpy as np

class UnweightedVectorMixtureModel():
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __getitem__(self, text):
        n = 0
        result = np.zeros(self.embedding_model.vector_size)
        for word in text:
            if word in self.embedding_model.wv:
                result += self.embedding_model.wv[word]
                n += 1
            
        return result / n


class TfIdfWeightedVectorMixtureModel():
    def __init__(self, embedding_model, tfidf_model, dictionary):
        self.embedding_model = embedding_model
        self.tfidf_model = tfidf_model
        self.dictionary = dictionary

    def __getitem__(self, text):
        n = 0
        result = np.zeros(self.embedding_model.vector_size)
        weights = self.tfidf_model[self.dictionary.doc2bow(text)]
        weights = dict(weights)
        for word in text:
            if word in self.embedding_model.wv:
                if word in self.dictionary.token2id:
                    weight = weights[self.dictionary.token2id[word]]
                else:
                    weight = 1
                result += weight * self.embedding_model.wv[word]
                n += weight
            
        return result / n
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

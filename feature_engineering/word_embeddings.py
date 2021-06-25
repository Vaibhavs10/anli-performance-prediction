import gensim
from gensim import downloader


def train_embedding_model(documents, type="word2vec", vector_size=300, min_count=100, skipgram=0):
    if type=="word2vec":
        return gensim.models.Word2Vec(sentences=documents, vector_size=vector_size, min_count=min_count, sg=skipgram)
    elif type=="fasttext":
        return gensim.models.FastText(sentences=documents, vector_size=vector_size, min_count=min_count, sg=skipgram)
    else:
        raise ValueError

def get_pretrained_embeddings(type):
    return EmbeddingProxy(gensim.downloader.load(type))

    
class EmbeddingProxy():
    def __init__(self, wv):
        self.wv = wv
        self.vector_size = wv.vector_size

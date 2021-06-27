import gensim
from gensim import downloader
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def train_embedding_model(documents, embedding_type, vector_size=300, min_count=10):
    if embedding_type == 'doc2vec':
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
        return Doc2Vec(documents, vector_size=vector_size, min_count=min_count)
    else:
        raise NotImplementedError

def get_pretrained_embeddings(type):
    raise NotImplementedError

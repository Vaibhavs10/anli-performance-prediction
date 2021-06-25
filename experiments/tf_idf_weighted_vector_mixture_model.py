from scipy import spatial
import importlib

import models.evaluation as evaluation
import feature_engineering.word_embeddings as word_embeddings
import feature_engineering.tf_idf as tf_idf
from models.similarity.vector_mixture_models import TfIdfWeightedVectorMixtureModel 

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
    
def _predict_labels(model, data):
    labels = []
    for instance in data:
        obs = model[instance[0] + instance[1]]
        hyp1 = model[instance[2]]
        hyp2 = model[instance[3]]

        s1 = spatial.distance.cosine(obs, hyp1)
        s2 = spatial.distance.cosine(obs, hyp2)
        if s1 < s2:
            labels.append('1')
        else:
            labels.append('2')
    return labels

def _load_data(dev_data_loader_name, embedding_data_loader_name, trace=False):
    embedding_training_corpus = None

    dev_data_loader = importlib.import_module("experiments." + dev_data_loader_name)
    dev_documents, _ = dev_data_loader.load_data()
    if embedding_data_loader_name:
        embedding_data_loader = importlib.import_module("experiments." + embedding_data_loader_name)
        _, embedding_training_corpus = embedding_data_loader.load_data()
    return dev_documents, embedding_training_corpus
       
def _get_param(dict, key, fallback_value):
    if key in dict:
        return dict[key]
    else:
        return fallback_value

def _run_vector_mixture_experiment(    
    use_pre_trained_embeddings,
    embedding_type,
    skipgram,
    dev_data_loader_name,
    embedding_training_data_loader_name,
    trace=True):

    logs=[]

    if trace:
        print("Loading data...")
    dev_documents, embedding_training_corpus = _load_data(dev_data_loader_name, embedding_training_data_loader_name, trace)
    if use_pre_trained_embeddings:
        if trace:
            print("Using pretrained embedding model %s ..." % embedding_type)
        embedding_model = word_embeddings.get_pretrained_embeddings(embedding_type)
    else:
        if trace:
            print("Training %s embedding model..." % embedding_type)
        embedding_model = word_embeddings.train_embedding_model(embedding_training_corpus, type=embedding_type, skipgram=skipgram)
        
    tfidf_model, dictionary = tf_idf.train_tf_idf_model(embedding_training_corpus)
    model = TfIdfWeightedVectorMixtureModel(embedding_model, tfidf_model, dictionary)
    if trace:
        print("Predicting labels on devset...")
    predicted_labels = _predict_labels(model, dev_documents)
    real_labels = [x[4] for x in dev_documents]
    acc = evaluation.calculate_accuracy(predicted_labels, real_labels)

    if trace:
        print("Accuracy: " + str(acc))
    logs.append("Accuracy: " + str(acc))

    return predicted_labels, logs


def run(ex):
    hp = ex["hyperparameters"]
    return _run_vector_mixture_experiment(
        use_pre_trained_embeddings=_get_param(hp, "pre_trained_embeddings", False),
        embedding_type=_get_param(hp, "embedding_type", "word2vec"),
        dev_data_loader_name=_get_param(hp, "dev_data_loader_name", "word2vec"),
        embedding_training_data_loader_name=_get_param(hp, "embedding_training_data_loader_name", "word2vec"),
        skipgram=_get_param(hp, "skipgram", False),
        trace=_get_param(ex, "trace", True),
        )

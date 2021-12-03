import time
import importlib
import models.evaluation as evaluation
import scipy.spatial as spatial
import feature_engineering.document_embeddings as document_embeddings
from experiments.experiment_utilities import get_param, printt, set_trace

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

def _load_data(dev_data_loader_name):
    """
    Dynamically load python file with the given name and call load_data_raw() on it.
    """

    dev_data_loader = importlib.import_module("experiments." + dev_data_loader_name)
    return dev_data_loader.load_data_raw()

def _predict_labels_cosine(model, data):
    """
    Use cosine similarity to make label predictions for each data point.
    Model needs to have an .encode() method that takes a list of non-tokenized texts.
    """
    obs_embeddings = model.encode([obs for obs, hyp1, hyp2, label in data])
    hyp1_embeddings = model.encode([hyp1 for obs, hyp1, hyp2, label in data])
    hyp2_embeddings = model.encode([hyp2 for obs, hyp1, hyp2, label in data])

    labels = []
    for i, obs_embedding in enumerate(obs_embeddings):
        hyp1_embedding = hyp1_embeddings[i]
        hyp2_embedding = hyp2_embeddings[i]
        s1 = spatial.distance.cosine(obs_embedding, hyp1_embedding)
        s2 = spatial.distance.cosine(obs_embedding, hyp2_embedding)
        if s1 < s2:
            labels.append('1')
        else:
            labels.append('2')

    return labels

def _run_transformer_similarity_experiment(    
    embedding_type,
    dev_data_loader_name):

    logs=[]
    printt("Starting similarity experiment for " + embedding_type + "...")
    printt("Loading data...")
    dev_documents = _load_data(dev_data_loader_name)

    transformer_model = document_embeddings.get_pretrained_embeddings(embedding_type)

    printt("Predicting labels on devset...")
    start = time.time()
    predicted_labels = _predict_labels_cosine(transformer_model, dev_documents)
    real_labels = [x[3] for x in dev_documents]
    acc = evaluation.calculate_accuracy(predicted_labels, real_labels)
    end = time.time()
    time_taken = end - start
    printt("Accuracy: " + str(acc))
    printt("Time taken: " + str(time_taken))
    logs.append("Accuracy: " + str(acc))
    logs.append("Time taken: " + str(time_taken))

    return predicted_labels, acc, logs


def run(ex):
    """
    Transformer sentence embeddings, compared using cosine similarity.
    Takes in a dictionary with all the hyperparameters, for example:
        {
            "trace": true,     

            "hyperparameters" : {
                "model_checkpoint":"bert-base-uncased",
                "dev_data_loader_name": "data_loaders.dev_data_loader"
            }
        }
    
    Returns: predicted labels, logs
    """  
    set_trace(get_param(ex, "trace", True))
    
    hp = ex["hyperparameters"]
    return _run_transformer_similarity_experiment(
        embedding_type=get_param(hp, "model_checkpoint", "word2vec"),
        dev_data_loader_name=get_param(hp, "dev_data_loader_name", None)
    )

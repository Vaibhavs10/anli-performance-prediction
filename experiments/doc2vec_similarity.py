import scipy.spatial as spatial
import models.evaluation as evaluation
import feature_engineering.document_embeddings as document_embeddings
from experiments.experiment_utilities import get_param, load_data, printt, set_trace

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

def _predict_labels(model, data):
    labels = []
    for instance in data:
        obs = model.infer_vector(instance[0] + instance[1])
        hyp1 = model.infer_vector(instance[2])
        hyp2 = model.infer_vector(instance[3])

        s1 = spatial.distance.cosine(obs, hyp1)
        s2 = spatial.distance.cosine(obs, hyp2)
        if s1 < s2:
            labels.append('1')
        else:
            labels.append('2')
    return labels    

def _run_doc2vec_experiment(    
    use_pre_trained_embeddings,
    embedding_type,
    dev_data_loader_name,
    embedding_training_data_loader_name,
    vector_size,
    min_count):

    logs=[]

    printt("Loading data...")
    dev_documents, embedding_training_corpus = load_data(dev_data_loader_name, embedding_training_data_loader_name)
    if use_pre_trained_embeddings:
        printt("Using pretrained embedding model %s ..." % embedding_type)
        embedding_model = document_embeddings.get_pretrained_embeddings(embedding_type)
    else:
        printt("Training %s embedding model..." % embedding_type)
        embedding_model = document_embeddings.train_embedding_model(embedding_training_corpus, embedding_type, vector_size, min_count)
        
    printt("Predicting labels on devset...")
    predicted_labels = _predict_labels(embedding_model, dev_documents)
    real_labels = [x[4] for x in dev_documents]
    acc = evaluation.calculate_accuracy(predicted_labels, real_labels)

    printt("Accuracy: " + str(acc))
    logs.append("Accuracy: " + str(acc))

    return predicted_labels, logs


def run(ex):
    """
    Doc2vec experiment with cosine similarity.
    Takes in a dictionary with all the hyperparameters, for example:
        {
            "trace": true,     

            "hyperparameters" : {
                "dev_data_loader_name": "data_loaders.dev_data_loader",
                "pre_trained_embeddings":false,
                "embedding_type":"word2vec",
                "embedding_training_data_loader_name": "data_loaders.ROC_data_loader",
                "vector_size":100,
                "min_count":10
            }
        }
    
    Returns: predicted labels, logs
    """  
    hp = ex["hyperparameters"]
    set_trace(get_param(ex, "trace", True))
    return _run_doc2vec_experiment(
        use_pre_trained_embeddings=get_param(hp, "pre_trained_embeddings", False),
        embedding_type=get_param(hp, "embedding_type", "doc2vec"),
        dev_data_loader_name=get_param(hp, "dev_data_loader_name", "word2vec"),
        embedding_training_data_loader_name=get_param(hp, "embedding_training_data_loader_name", "word2vec"),
        vector_size=get_param(hp, "vector_size", 100),
        min_count=get_param(hp, "min_count", 10),
        )

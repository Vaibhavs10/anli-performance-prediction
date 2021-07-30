import models.evaluation as evaluation
import feature_engineering.word_embeddings as word_embeddings
from models.similarity.vector_mixture_models import UnweightedVectorMixtureModel 
from experiments.experiment_utilities import get_param, predict_labels_cosine, load_data, printt, set_trace

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

def _run_vector_mixture_experiment(    
    use_pre_trained_embeddings,
    embedding_type,
    vector_size,
    min_count,
    skipgram,
    dev_data_loader_name,
    embedding_training_data_loader_name):

    logs=[]

    printt("Loading data...")
    dev_documents, embedding_training_corpus = load_data(dev_data_loader_name, embedding_training_data_loader_name)

    if use_pre_trained_embeddings:
        printt("Using pretrained embedding model %s ..." % embedding_type)
        embedding_model = word_embeddings.get_pretrained_embeddings(embedding_type)
    else:
        printt("Training %s embedding model..." % embedding_type)
        embedding_model = word_embeddings.train_embedding_model(
            embedding_training_corpus, 
            type=embedding_type, 
            vector_size=vector_size, 
            min_count=min_count, 
            skipgram=skipgram
            )
        
    model = UnweightedVectorMixtureModel(embedding_model)

    printt("Predicting labels on devset...")
    predicted_labels = predict_labels_cosine(model, dev_documents)
    real_labels = [x[4] for x in dev_documents]

    acc = evaluation.calculate_accuracy(predicted_labels, real_labels)
    
    printt("Accuracy: " + str(acc))
    logs.append("Accuracy: " + str(acc))

    return predicted_labels, acc, logs


def run(ex):
    """
    Unweighted vector mixture experiments.
    Takes in a dictionary with all the hyperparameters, for example:
        {
            "trace": true,     
            "hyperparameters" : {
                "dev_data_loader_name": "data_loaders.dev_data_loader",
                "pre_trained_embeddings":false,
                "embedding_type":"word2vec",
                "embedding_training_data_loader_name": "data_loaders.ROC_data_loader",
                "skipgram":true
            }
        }
    
    Returns: predicted labels, logs
    """
    set_trace(get_param(ex, "trace", True))

    hp = ex["hyperparameters"]
    return _run_vector_mixture_experiment(
        use_pre_trained_embeddings=get_param(hp, "pre_trained_embeddings", False),
        embedding_type=get_param(hp, "embedding_type", "word2vec"),
        vector_size=get_param(hp, "vector_size", 100),
        min_count=get_param(hp, "min_count", 10),
        skipgram=get_param(hp, "skipgram", False),
        dev_data_loader_name=get_param(hp, "dev_data_loader_name", None),
        embedding_training_data_loader_name=get_param(hp, "embedding_training_data_loader_name", "word2vec")
    )

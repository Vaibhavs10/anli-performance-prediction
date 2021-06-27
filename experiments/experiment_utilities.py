import importlib
import scipy.spatial as spatial

g_trace = False
def printt(text):
    global g_trace
    if g_trace:
        print(text)

def set_trace(trace_value):
    global g_trace
    g_trace = trace_value

def get_param(dict, key, fallback_value):
    """
    Get parameter with the provided key from the dictionary. If it doesn't exist, use fallback_value.
    """
    if key in dict:
        return dict[key]
    else:
        return fallback_value


def load_data(dev_data_loader_name, embedding_data_loader_name):
    """
    Dynamically load python files with the given names and call load_data() on them.
    These data loaders take care of finding the data and tokenizing.
    """
    embedding_training_corpus = None
    dev_documents = None

    # load dev data
    dev_data_loader = importlib.import_module("experiments." + dev_data_loader_name)
    dev_documents, _ = dev_data_loader.load_data()

    # if data loader for training data is given, load it. This is not needed for pre-trained embeddings
    if embedding_data_loader_name:
        embedding_data_loader = importlib.import_module("experiments." + embedding_data_loader_name)
        _, embedding_training_corpus = embedding_data_loader.load_data()

    return dev_documents, embedding_training_corpus

def predict_labels_cosine(model, data):
    """
    Use cosine similarity to make label predictions for each data point.
    Model must override __getitem__ to take in a text and return the vector representation of the whole text. 
    """
    labels = []
    for instance in data:
        # combine observations
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

def predict_labels_wmd(model, data):
    labels = []
    for instance in data:
        obs = instance[0] + instance[1]
        hyp1 = instance[2]
        hyp2 = instance[3]
        
        s1 = model.wv.wmdistance(obs, hyp1)
        s2 = model.wv.wmdistance(obs, hyp2)
        if s1 < s2:
            labels.append('1')
        else:
            labels.append('2')
    return labels
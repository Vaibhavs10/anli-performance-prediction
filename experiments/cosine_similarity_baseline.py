import json
import random

from models.similarity.cosine_similarity import cosine_similarity
from feature_engineering import vectorizer
from data import data_loader
from models import evaluation

from experiments.experiment_utilities import get_param

# Experiment1: Cosine similarity between averaged obs1, obs2 vectors and hypothesis vectors

def _run_cosine_similarity_exp(train_file_path, test_file_path, ):
    train_list_of_rows = data_loader.parse_and_return_rows(train_file_path)
    test_list_of_rows = data_loader.parse_and_return_rows(test_file_path)
    train_vocab, train_len_vocab = vectorizer.return_len_and_vocabulary(train_list_of_rows)
    index_word = vectorizer.create_token_index(train_vocab)
    hyp1_sim = []
    hyp2_sim = []
    count = 0

    for row in test_list_of_rows:
        obs1 = vectorizer.preprocess_sentence(row[1])
        obs1_vec = vectorizer.return_count_vector(obs1, index_word, train_len_vocab)
        obs2 = vectorizer.preprocess_sentence(row[2])
        obs2_vec = vectorizer.return_count_vector(obs2, index_word, train_len_vocab)
        hyp1 = vectorizer.preprocess_sentence(row[3])
        hyp1_vec = vectorizer.return_count_vector(hyp1, index_word, train_len_vocab)
        hyp2 = vectorizer.preprocess_sentence(row[4])
        hyp2_vec = vectorizer.return_count_vector(hyp2, index_word, train_len_vocab)

        avg_obs_vec = (obs1_vec + obs2_vec)/2

        hyp1_sim.append(cosine_similarity(avg_obs_vec, hyp1_vec))
        hyp2_sim.append(cosine_similarity(avg_obs_vec, hyp2_vec))

        count+=1

    return hyp1_sim, hyp2_sim

def _calc_test_similarity_accuracy(test_file_path, hyp1_sim, hyp2_sim):

    pred_list = []

    for i in range(len(hyp1_sim)):
        if hyp1_sim[i] > hyp2_sim[i]:
            pred_list.append("1")
        elif hyp1_sim[i] < hyp2_sim[i]:
            pred_list.append("2")
        elif hyp1_sim[i] == hyp2_sim[i]:
            rand_choice = random.choice(["1","2"])
            pred_list.append(rand_choice)
    
    test_list_of_rows = data_loader.parse_and_return_rows(test_file_path)
    orig_list = []
    for row in test_list_of_rows:
        orig_list.append(str(row[5]))

    accuracy = evaluation.calculate_accuracy(pred_list, orig_list)
    return accuracy, pred_list

def _run_cosine_similarity_baseline(train_file_path, test_file_path):

    hyp1_sim, hyp2_sim = _run_cosine_similarity_exp(train_file_path, test_file_path)
    accuracy, pred_list = _calc_test_similarity_accuracy(test_file_path, hyp1_sim, hyp2_sim)

    return pred_list, accuracy, []

def run(ex):
    hp = ex["hyperparameters"]

    return _run_cosine_similarity_baseline(train_file_path=get_param(hp, "train_file_path", None),
        test_file_path=get_param(hp, "test_file_path", None))
import json
import random

from models.similarity.cosine_similarity import cosine_similarity
from feature_engineering import vectorizer
from data import data_loader
from models import evaluation

# Experiment1: Cosine similarity between averaged obs1, obs2 vectors and hypothesis vectors

def _run_cosine_similarity_exp(train_file_path, test_file_path):
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

        print(count)

    with open("hyp1_sim.txt", "w") as fp:
        json.dump(hyp1_sim, fp)

    with open("hyp2_sim.txt", "w") as fp:
        json.dump(hyp2_sim, fp)

def _calc_test_similarity_accuracy(test_file_path, hyp1_sim_file_path, hyp2_sim_file_path):

    with open(hyp1_sim_file_path, "r") as fp:
        hyp1_sim = json.load(fp)

    with open(hyp2_sim_file_path, "r") as fp:
        hyp2_sim = json.load(fp)

    pred_list = []

    for i in range(len(hyp1_sim)):
        if hyp1_sim[i] > hyp2_sim[i]:
            pred_list.append("1")
        elif hyp1_sim[i] < hyp2_sim[i]:
            pred_list.append("2")
        elif hyp1_sim[i] == hyp2_sim[i]:
            rand_choice = random.choice(["1","2"])
            pred_list.append(rand_choice)
    
    test_list_of_rows = vectorizer.parse_and_return_rows(test_file_path)
    orig_list = []
    for row in test_list_of_rows:
        orig_list.append(str(row[5]))

    accuracy = evaluation.calculate_accuracy(pred_list, orig_list)
    return accuracy

def _run_tfidf_similarity_exp(train_file_path, test_file_path):
    list_of_rows = data_loader.parse_and_return_rows(train_file_path)
    test_list_of_rows = data_loader.parse_and_return_rows(test_file_path)
    obs1, obs2, hyp1, hyp2 = vectorizer.return_tfidf_row_lists(test_list_of_rows)
    corpus = vectorizer.create_tfidf_corpus(list_of_rows)
    vect = vectorizer.fit_tfidf_vectorizer(corpus)
    
    hyp1_sim = []
    hyp2_sim = []
    
    obs1_vec = vectorizer.transform_tfidf_vectorizer(vect, obs1).toarray()
    obs2_vec = vectorizer.transform_tfidf_vectorizer(vect, obs2).toarray()
    hyp1_vec = vectorizer.transform_tfidf_vectorizer(vect, hyp1).toarray()
    hyp2_vec = vectorizer.transform_tfidf_vectorizer(vect, hyp2).toarray()

    print(len(obs1_vec))
    print(len(obs2_vec))
    print(len(hyp1_vec))
    print(len(hyp2_vec))

    for i in range(len(obs1_vec)):
        avg_obs_vec = (obs1_vec[i] + obs2_vec[i])/2
        hyp1_sim.append(cosine_similarity(avg_obs_vec, hyp1_vec[i]))
        hyp2_sim.append(cosine_similarity(avg_obs_vec, hyp2_vec[i]))

    print(len(hyp1_vec))
    print(len(hyp2_vec))

    with open("hyp1_tfidf_sim.txt", "w") as fp:
        json.dump(hyp1_sim, fp)
    
    with open("hyp2_tfidf_sim.txt", "w") as fp:
        json.dump(hyp2_sim, fp)

        
def run(ex):
    train_file_path = "data/processed_data/train.csv"
    test_file_path = "data/processed_data/dev.csv"

    hyp1_sim_file_path = "hyp1_sim.txt"
    hyp2_sim_file_path = "hyp2_sim.txt"

    hyp1_tfidf_sim_file_path = "hyp1_tfidf_sim.txt"
    hyp2_tfidf_sim_file_path = "hyp2_tfidf_sim.txt"

    #run_tfidf_similarity_exp(train_file_path, test_file_path)
    #accuracy = calc_test_similarity_accuracy(test_file_path, hyp1_tfidf_sim_file_path, hyp2_tfidf_sim_file_path)
    #print(accuracy)
    _run_cosine_similarity_exp(train_file_path, test_file_path)
    accuracy = _calc_test_similarity_accuracy(test_file_path, hyp1_sim_file_path, hyp2_sim_file_path)
    print(accuracy)
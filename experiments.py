import json

from utils import similarity
from utils import vectorizer

# Experiment1: Cosine similarity between averaged obs1, obs2 vectors and hypothesis vectors

file_path = "utils/data/processed_data/train.csv"

def run_cosine_similarity_exp(file_path):
    list_of_rows = vectorizer.parse_and_return_rows(file_path)
    vocab, len_vocab = vectorizer.return_len_and_vocabulary(list_of_rows)
    index_word = vectorizer.create_token_index(vocab)
    hyp1_sim = []
    hyp2_sim = []
    for row in list_of_rows:
        obs1 = vectorizer.preprocess_sentence(row[1])
        print(obs1)
        obs1_vec = vectorizer.return_count_vector(obs1, index_word, len_vocab)
        obs2 = vectorizer.preprocess_sentence(row[2])
        print(obs2)
        obs2_vec = vectorizer.return_count_vector(obs2, index_word, len_vocab)
        hyp1 = vectorizer.preprocess_sentence(row[3])
        print(hyp1)
        hyp1_vec = vectorizer.return_count_vector(hyp1, index_word, len_vocab)
        hyp2 = vectorizer.preprocess_sentence(row[4])
        print(hyp2)
        hyp2_vec = vectorizer.return_count_vector(hyp2, index_word, len_vocab)

        avg_obs_vec = (obs1_vec + obs2_vec)/2
        print("-_________-")
        print(similarity.cosine_similarity(avg_obs_vec, hyp1_vec))
        print(similarity.cosine_similarity(avg_obs_vec, hyp2_vec))
        hyp1_sim.append(similarity.cosine_similarity(avg_obs_vec, hyp1_vec))
        hyp2_sim.append(similarity.cosine_similarity(avg_obs_vec, hyp2_vec))

        with open("hyp1_sim.txt", "w") as fp:
            json.dump(hyp1_sim, fp)

        with open("hyp2_sim.txt", "w") as fp:
            json.dump(hyp2_sim, fp)

run_cosine_similarity_exp(file_path)

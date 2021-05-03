from utils import similarity
from utils import vectorizer

# Experiment1: Cosine similarity between averaged obs1, obs2 vectors and hypothesis vectors

file_path = "utils/data/processed_data/train.csv"

list_of_rows = vectorizer.parse_and_return_rows(file_path)
vocab, len_vocab = vectorizer.return_len_and_vocabulary(list_of_rows)
index_word = vectorizer.create_token_index(vocab)
test = vectorizer.preprocess_sentence("Chad is a boy")
print(vectorizer.return_count_vector(test, index_word, len_vocab))

def run_cosine_similarity_exp(file_path):
    list_of_rows = vectorizer.parse_and_return_rows(file_path)
    vocab, len_vocab = vectorizer.return_len_and_vocabulary(list_of_rows)
    index_word = vectorizer.create_token_index(vocab)
    test = vectorizer.preprocess_sentence("Chad is a boy")
    test = vectorizer.return_count_vector(test, index_word, len_vocab)
    print(similarity.cosine_similarity(test, test))

run_cosine_similarity_exp(file_path)

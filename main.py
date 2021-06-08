import sys

from experiments.decision_tree_baseline import run_decision_tree_experiment
from experiments.cosine_similarity_baseline import run_cosine_similarity_exp, run_tfidf_similarity_exp, calc_test_similarity_accuracy
from experiments.human_baseline import run_human_baseline_experiment

def run_decision_tree_baseline_experiments():
    run_decision_tree_experiment(
        max_depth=100,
        subset_size=3000,
        feature_removal_threshold=100,
        training_instance_threshold=100,
        result_file_name='experiment_results/deicision_tree_baseline.txt',
        num_threads=5,
        print_logs=True,
        accuracy_print_frequency=10
    )


def run_cosine_similarity_baseline_experiments():
    train_file_path = "data/processed_data/train.csv"
    test_file_path = "data/processed_data/dev.csv"

    hyp1_sim_file_path = "hyp1_sim.txt"
    hyp2_sim_file_path = "hyp2_sim.txt"

    hyp1_tfidf_sim_file_path = "hyp1_tfidf_sim.txt"
    hyp2_tfidf_sim_file_path = "hyp2_tfidf_sim.txt"

    #run_tfidf_similarity_exp(train_file_path, test_file_path)
    #accuracy = calc_test_similarity_accuracy(test_file_path, hyp1_tfidf_sim_file_path, hyp2_tfidf_sim_file_path)
    #print(accuracy)
    run_cosine_similarity_exp(train_file_path, test_file_path)
    accuracy = calc_test_similarity_accuracy(test_file_path, hyp1_sim_file_path, hyp2_sim_file_path)
    print(accuracy)

if __name__ == "__main__":
    #run_decision_tree_baseline_experiments()
    #run_cosine_similarity_baseline_experiments()
    run_human_baseline_experiment(100)

    

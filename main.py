import json
import argparse
# setting up command line interface
parser = argparse.ArgumentParser(description='Various experiments for abductive natural language inferrence.')
parser.add_argument('--infile', nargs=1,
                    help="JSON file containing details about which experiment to run",
                    type=argparse.FileType('r'))
args = parser.parse_args()

from experiments.decision_tree_baseline import run_decision_tree_experiment
from experiments.cosine_similarity_baseline import run_cosine_similarity_exp, run_tfidf_similarity_exp, calc_test_similarity_accuracy
from experiments.human_baseline import run_human_baseline_experiment

def get_param(dict, key, fallback_value):
    if key in dict:
        return dict[key]
    else:
        return fallback_value

def run_decision_tree_baseline_experiment(ex):
    hp = ex["hyperparameters"]
    run_decision_tree_experiment(
        max_depth=get_param(hp, "max_depth", 100),
        subset_size=get_param(hp, "subset_size", 3000),
        feature_removal_threshold=get_param(hp, "feature_removal_threshold", 100),
        training_instance_threshold=get_param(hp, "training_instance_threshold", 100),
        result_file_name=get_param(ex, "result_file_name", 'result.txt'),
        num_threads=get_param(ex, "num_threads", 5),
        print_logs=get_param(ex, "trace", True),
        accuracy_print_frequency=get_param(ex, "accuracy_print_frequency", 10),
    )


def run_cosine_similarity_baseline_experiments(ex):
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
    if args.infile is not None and args.infile[0] is not None:
        experiment_definition = json.load(args.infile[0])
        id = experiment_definition['experiment_id']
        if id == "decision_tree_baseline":
            run_decision_tree_baseline_experiment(experiment_definition)
        elif id == "cosine_similarity_baseline":
            run_cosine_similarity_baseline_experiments(experiment_definition)
    else:
        print("Please pass in a .json file defining the experiment to run with --infile <file>")
        input("Press Enter to run a human baseline...")
        run_human_baseline_experiment(100)

    

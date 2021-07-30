import json
import importlib
import argparse
import time
import os
from itertools import product
# setting up command line interface
parser = argparse.ArgumentParser(description='Various experiments for abductive natural language inferrence.')
parser.add_argument('--infile', nargs=1,
                    help="JSON file containing details about which experiment to run",
                    type=argparse.FileType('r'))

# get arguments that were passed through command line
args = parser.parse_args()

from experiments.human_baseline import run_human_baseline_experiment


def save_results(experiment_definition, labels, accuracy, logs):
    """
    Saves the results in the ./experiment_results directory in a folder named <acc>_<experiment>_<timestamp>.
    Saves 3 files: 
        1) experiment.json, which contains all the hyperparameters used for the experiment
        2) labels.lst, which contains all the predicted labels on the dev set
        3) log.txt, which contains any log entries that the experiment may have returned
    """

    # create folder for results
    id = experiment_definition['experiment_id']
    timestamp = time.strftime("%m_%d__%H_%M_%S", time.gmtime())
    base_path = os.path.join('experiment_results', str(accuracy) + "_" + id + '_' + timestamp)
    os.mkdir(base_path)

    # save hyperparameters
    file_name = "experiment.json"
    file = open(os.path.join(base_path, file_name), "w")
    file.writelines(json.dumps(experiment_definition, indent=4))
    file.close()

    # save labels
    if labels:
        file_name = "labels.lst"
        file = open(os.path.join(base_path, file_name), "w")
        for l in labels:
            file.write(l + '\n')
        file.close()

    # save logs, if any were returned
    if logs:
        file_name = "log.txt"
        file = open(os.path.join(base_path, file_name), "w")
        file.writelines(logs)
        file.close()

def product_dict(**kwargs):
    """
    Given a dictionary such as {a: [1,2,3], b: [4,5,6]}
    Iterates through all possbiel combinations: [{a: 1, b: 4}, {a: 1, b: 5}, {a: 1, b: 6}, ....]

    Code taken from: https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def expand_experiments(experiments):
    """
    For every experiment that has 'try_combinations': [column_a, column_b, ...] defined, 
    expands it into multiple separate experiments, trying all possible combinations of the values given in column_a, column_b, ....
    """
    expanded_experiments = []
    for experiment_definition in experiments:
        if 'try_combinations' in experiment_definition:
            column_names = experiment_definition['try_combinations']
            all_possible_values = {column_name: experiment_definition['hyperparameters'][column_name] for column_name in column_names}
            all_possible_value_combinations = product_dict(**all_possible_values)

            # create a new experiment definition for each combination
            for combination in all_possible_value_combinations:
                new_experiment_definition = experiment_definition.copy()
                new_experiment_definition['hyperparameters'] = experiment_definition['hyperparameters'].copy()

                new_experiment_definition.pop('try_combinations')
                # change the value of each hyperparameter that are being varied
                for key, value in combination.items():
                    new_experiment_definition['hyperparameters'][key] = value
                expanded_experiments.append(new_experiment_definition)
        else:
            expanded_experiments.append(experiment_definition)
    return expanded_experiments



if __name__ == "__main__":
    #args.infile = [open('wmd_similarity.json', 'r')] # uncomment this to run an experiment without having to pass it in the command line
    if args.infile is not None and args.infile[0] is not None:
        # load the passed json file that contains details about the experiment to run
        all_experiments = json.load(args.infile[0])['experiments']
        all_experiments = expand_experiments(all_experiments)
        for experiment_definition in all_experiments:
            id = experiment_definition['experiment_id']
            try:
                # load the appropriate .py file from the experiments folder
                experiment = importlib.import_module("experiments." + id)
            except ModuleNotFoundError:
                print("Experiment at experiments.%s not found" % id)
                continue
            labels, accuracy, logs = experiment.run(experiment_definition)

            # some experiments take care of saving their results themselves. For example, decision trees calculate accuracy at ALL depths, so the result saving is different
            if labels is not None:
                save_results(experiment_definition, labels, accuracy, logs)

    else:
        print("Please pass in a .json file defining the experiment to run with --infile <file>")
        input("Press Enter to run a human baseline...")
        run_human_baseline_experiment(100)

    

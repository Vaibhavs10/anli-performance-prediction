import json
import importlib
import argparse
import time
import os
# setting up command line interface
parser = argparse.ArgumentParser(description='Various experiments for abductive natural language inferrence.')
parser.add_argument('--infile', nargs=1,
                    help="JSON file containing details about which experiment to run",
                    type=argparse.FileType('r'))

# get arguments that were passed through command line
args = parser.parse_args()

from experiments.human_baseline import run_human_baseline_experiment


def save_results(experiment_definition, labels, logs):
    id = experiment_definition['experiment_id']
    timestamp = time.strftime("%m_%d__%H_%M_%S", time.gmtime())
    base_path = os.path.join('experiment_results', id + '_' + timestamp)
    os.mkdir(base_path)

    file_name = "experiment.json"
    file = open(os.path.join(base_path, file_name), "w")
    file.writelines(json.dumps(experiment_definition, indent=4))
    file.close()

    file_name = "labels.lst"
    file = open(os.path.join(base_path, file_name), "w")
    for l in labels:
        file.write(l + '\n')
    file.close()

    if logs:
        file_name = "log.txt"
        file = open(os.path.join(base_path, file_name), "w")
        file.writelines(logs)
        file.close()


if __name__ == "__main__":
    #args.infile = [open('wmd_similarity.json', 'r')] # uncomment this to run an experiment without having to pass it in the command line
    if args.infile is not None and args.infile[0] is not None:
        # load the passed json file that contains details about the experiment to run
        all_experiments = json.load(args.infile[0])['experiments']
        for experiment_definition in all_experiments:
            id = experiment_definition['experiment_id']
            try:
                # load the appropriate .py file from the experiments folder
                experiment = importlib.import_module("experiments." + id)
            except ModuleNotFoundError:
                print("Experiment at experiments.%s not found" % id)
                continue
            labels, logs = experiment.run(experiment_definition)
            if labels is not None:
                save_results(experiment_definition, labels, logs)

    else:
        print("Please pass in a .json file defining the experiment to run with --infile <file>")
        input("Press Enter to run a human baseline...")
        run_human_baseline_experiment(100)

    

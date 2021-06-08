from csv import reader
from collections import defaultdict 
import random

import time

from feature_engineering import vectorizer, word_incidence_features
from models.classifiers.decision_tree import BinaryDecisionTree
from models import evaluation
from data import data_loader


def _load_data(feature_removal_threshold=None, print_logs=True):
    rows = data_loader.parse_and_return_rows('./data/processed_data/train.csv')
    dev_rows = data_loader.parse_and_return_rows('./data/processed_data/dev.csv')
    rows, dev_rows = data_loader.resample_dataset(rows, dev_rows)
    vocabulary, vocabulary_length = vectorizer.return_len_and_vocabulary(rows)
    word_to_index = vectorizer.create_token_index(vocabulary)
    x, y, total_feature_amount = word_incidence_features.features3(rows, word_to_index)
    x_dev, y_dev, _ = word_incidence_features.features3(dev_rows, word_to_index)

    if feature_removal_threshold is not None:
        x, y, x_dev, y_dev = word_incidence_features.prune_rare_features(x, y, total_feature_amount, feature_removal_threshold, x_dev=x_dev, y_dev=y_dev)

    if print_logs:
        print(len(x), " instances, ", total_feature_amount, " features")
        print(len(x_dev), " test instances")

    return x, y, x_dev, y_dev

def _train_tree(
    x, y, x_dev, y_dev,
    max_depth, 
    subset_size, 
    training_instance_threshold, 
    num_threads=1, 
    print_logs=True, 
    accuracy_print_frequency=10):
    """
    Performs training of the decision tree one step at a time, while recording various statistics about the process.
    """
    start = time.time()

    decision_tree = BinaryDecisionTree()
    decision_tree.initialize_training(x, y)

    logs = []
    can_keep_expanding = True

    while can_keep_expanding:
        if max_depth is not None and decision_tree.current_depth >= max_depth:
            return logs, decision_tree # max depth reached, stop training early
        
        one_layer_start = time.time()

        # expand the next depth layer of the tree
        can_keep_expanding = decision_tree.expand_tree(subset_size, training_instance_threshold, num_threads)

        one_layer_end = time.time()
        
        layer_time = one_layer_end - one_layer_start
        total_time = one_layer_end - start

        if print_logs:
            print("Depth: ", decision_tree.current_depth, "Total nodes: ", decision_tree.total_nodes, "Time taken on layer: ", layer_time, "Total time taken: ", total_time)
        logs.append((decision_tree.current_depth, decision_tree.total_nodes, layer_time, total_time))

        # Calculating accuracy (especially on the training set) is a bit expensive, so we're not doing it every step.
        # This is only meant to check up on the progress, since the final accuracy will be calcuated at all depths simultaneously after training is done.
        if print_logs and decision_tree.current_depth % accuracy_print_frequency == 0:
            print("dev: ", _calculate_accuracy(decision_tree, x_dev, y_dev))
            print("train: ", _calculate_accuracy(decision_tree, x, y))

    return logs, decision_tree

def _calculate_accuracy(decision_tree, x, y):
    """
    Make a prediction on each instance in x with the given decision_tree, then calculate accuracy by comparing predictions to the labels in y.
    """
    predictions = []
    for instance in x:
        predictions.append(decision_tree.predict_class(instance))
    return evaluation.calculate_accuracy(predictions, y)

def _calculate_accuracy_at_all_depths(decision_tree, x, y):
    """
    Make predictions on all instances in x with the given decision_tree at each depth.
    Returns a list of accuracies, where accuracies[i] is the accuracy that would be obtained if traversing the tree was stopped when it reached a depth of i.
    """
    accuracies = []

    # make predictions on each instance of x. We get a list of lists of size len(x) * depth
    predictions_at_all_depths = decision_tree.predict_classes_at_all_depths(x)

    # calculate transpose of predictions, so that it becomes depth * len(x)
    predictions_at_all_depths = list(map(list, zip(*predictions_at_all_depths)))

    # now when iterating through the predictions each row is a list of predicted classes for all instances in x
    for predictions_at_depth in predictions_at_all_depths:
        accuracy = evaluation.calculate_accuracy(predictions_at_depth, y)
        accuracies.append(accuracy)

    return accuracies

def _save_results(logs, train_accuracies, dev_accuracies, file_name):
    """
    Saves training statistics in a file with the given name.
    """
    file = open(file_name, "w")
    file.write("acc_train,acc_dev,depth,nodes,time_for_layer,total_time\n")
    for i, log in enumerate(logs):
        message = "%s,%s,%s,%s,%s,%s\n" % (train_accuracies[i], dev_accuracies[i], log[0], log[1], log[2], log[3])
        file.write(message)
    file.close()

def run_decision_tree_experiment(
    max_depth,
    subset_size,
    feature_removal_threshold,
    training_instance_threshold,
    result_file_name,
    num_threads=1,
    print_logs=True,
    accuracy_print_frequency=10):
    """
    Combines all the previous functions in a self contained test. Trains decision tree, calculates accuracies and saves the results.

    max_depth: when the tree reaches this depth, training will be stopped
    subset_size: how many training instances to look at when calculating information depth. Using only a subset speeds up training at the cost of accuracy.
    training_instance_threshold: normalizing measure. Don't split leaf nodes that have less than this amount of training instances associated with them.
    result_file_name: file name in which to store results
    num_threads: how many processes to use when calculating information gain. 5 works well on my computer, but will depend on hardware.
    accuracy_print_frequency: how often to calculate the accuracy and print it out. Only used for checking on the progress of the tree, full accuracy will be                                 calculated at the end anyway.
    tree, logs: in case the training is stopped before it's finished, if you passed your own tree/logs instances here, you'll still be able to access them 
                so the training progress won't be lost.
    """
    x, y, x_dev, y_dev = _load_data(feature_removal_threshold, print_logs)
    logs, tree = _train_tree(
        x, y, x_dev, y_dev, 
        max_depth, 
        subset_size, 
        training_instance_threshold, 
        num_threads, 
        print_logs, 
        accuracy_print_frequency)

    if print_logs:                      
        print("Calculating accuracy at all depths...")
    start = time.time()
    train_accuracies = _calculate_accuracy_at_all_depths(tree, x, y)
    dev_accuracies = _calculate_accuracy_at_all_depths(tree, x_dev, y_dev)
    end = time.time()
    if print_logs:                      
        print("Time taken: ", end - start)
        print("Final train accuracy: ", train_accuracies[-1])
        print("Final dev accuracy: ", dev_accuracies[-1])
    _save_results(logs, train_accuracies, dev_accuracies, result_file_name)
    return logs, tree

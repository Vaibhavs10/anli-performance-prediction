from multiprocessing import Pool
import math
from collections import Counter
from operator import itemgetter
import random

# Pool of processes to use in calculations. 
# Needs to be global, because creating a new one every time it's needed is expensive.
pool = None

class BinaryDecisionTree():
    """
    A wrapper class for BinaryDecisionTreeNode. Takes care of training and making predictions with the tree.
    """
    def __init__(self):
        self.root = None
        self.total_nodes = 0
        self.current_depth = 0

    def initialize_training(self, x, y):
        """
        Prepare the tree to start training it 
        """
        self.root = BinaryDecisionTreeNode(x, y)
        self.leaf_nodes_to_split = [self.root]
        self.new_leaf_nodes = []
        self.total_nodes = 1

    def expand_tree(self, subset_for_information_gain_calculation=None, num_threads=1):
        """
        Perform expansion of all nodes in self.leaf_nodes_to_split - all nodes in a single layer.
        Returns true if further expansion is possible, false if not.
        """
        while len(self.leaf_nodes_to_split) > 0:
            node = self.leaf_nodes_to_split.pop()
            node.split(subset_for_information_gain_calculation, num_threads)
            for child in node.children.values():
                self.new_leaf_nodes.append(child)
                self.total_nodes += 1

        # no new nodes were added, so we return false and stop
        if len(self.new_leaf_nodes) == 0:
            return False

        self.leaf_nodes_to_split = self.new_leaf_nodes
        self.new_leaf_nodes = []
        self.current_depth += 1 
        return True

    def train(self, max_depth=None, subset_for_information_gain_calculation=None):
        """
        Keep expanding tree until either no more nodes can be expanded or given max_depth is reached.
        """
        can_keep_expanding = True
        while can_keep_expanding:
            if max_depth is not None and self.current_depth >= max_depth:
                return
            can_keep_expanding = self.expand_tree(subset_for_information_gain_calculation)

    def predict_class(self, x):
        """
        Make prediction on a single instance.
        """
        return self.root.predict_class(x)

    def predict_classes_at_all_depths(self, instances):    
        """
        Dynamically predict the class of the tree on the given instances at every depth.
        Returns a (len(instances) x max_depth) array with the predicted classes. 
        """    
        predictions_for_all_instances = []

        # calculate class at each depth for every instance x
        for x in instances:
            # start at the root
            current_node = self.root

            # prediction at depth 0 is just the most common classes in the training data
            predictions = [current_node.most_common_class]

            # traverse the tree until a leaf is hit, saving the most_common_class at every step
            while not current_node.is_leaf:
                feature_value = 1 if current_node.split_feature_index in x else 0
                current_node = current_node.children[feature_value]
                predictions.append(current_node.most_common_class)

            # store predictions for this x in the 
            predictions_for_all_instances.append(predictions)

        # extend all the arrays for the predictions that terminated before reaching max depth, 
        # so that each test instance has exactly max_depth predictions
        for i, prediction in enumerate(predictions_for_all_instances):
            prediction += [prediction[-1]] * (self.current_depth + 1 - len(prediction))
            predictions_for_all_instances[i] = prediction

        return predictions_for_all_instances


class BinaryDecisionTreeNode():
    """One node in a decision tree that can split into 2 branches"""

    def __init__(self, x, y, already_used_features = [], most_common_class_of_parent=None):
        self.x = x
        self.y = y
        self.already_used_features = already_used_features  # indexes of all the features that have already been used to split the tree
        self.is_leaf = True
        if len(y) == 0:
            self.entropy = 0        
            self.most_common_class = most_common_class_of_parent
        else:
            self.entropy = entropy(self.y)        
            self.most_common_class = Counter(self.y).most_common()[0][0]
        self.split_feature_index = None                     # the index of the feature that this node splits on
        self.children = {}


    def split(self, subset_for_information_gain_calculation=None, num_threads=1):
        """Split this into two on the feature that would lead to the biggest information gain"""
        info_gain_for_all_features = information_gain(self.x, self.y, feature_mask=self.already_used_features, subset_for_information_gain_calculation=subset_for_information_gain_calculation, num_threads=num_threads)
        # if no possible split leads to an information gain, don't split
        if len(info_gain_for_all_features) == 0 or max(info_gain_for_all_features.values()) == 0:            
            return

        self.is_leaf = False
        self.split_feature_index = dictionary_argmax(info_gain_for_all_features) # split on the feature that leads to the most information gain
        good_x, bad_x, good_y, bad_y = split_on_binary_feature(self.x, self.y, self.split_feature_index)

        new_used_features = self.already_used_features + [self.split_feature_index]
        self.children[0] = BinaryDecisionTreeNode(bad_x, bad_y, new_used_features, self.most_common_class)
        self.children[1] = BinaryDecisionTreeNode(good_x, good_y, new_used_features, self.most_common_class) 

    def split_recursively(self, depth=0, max_depth=None):
        """Keep splitting recursively from this node"""  
        # Entropy == 0 when all the instances in this node have the same class as there is no uncertainty anymore
        if self.entropy == 0:        
            return

        if max_depth is not None and depth > max_depth:
            return
        
        self.split()

        for _, child in self.children.items():
            child.split_recursively(depth=depth+1, max_depth=max_depth)

    def predict_class(self, x):
        """Recursively make a prediction about an unseen instance"""
        if self.is_leaf:
            return self.most_common_class
        else:
            feature_value = 1 if self.split_feature_index in x else 0
            return self.children[feature_value].predict_class(x) 

def argmax(array):
      return array.index(max(array))

def dictionary_argmax(dictionary):
    return max(dictionary.items(), key=itemgetter(1))[0]

def entropy(y):
    """Calculate entropy from list of labels in the dataset"""
    class_counter = Counter(y)
    entropy = 0
    for c in class_counter:
        probability = class_counter[c] / len(y)
        entropy += -probability * math.log2(probability)
    return entropy

def split_on_binary_feature(x, y, feature_index):
    """Split instances x and labels y into two sets based on whether the feature at feature_index is 1 or 0"""
    good_x, bad_x, good_y, bad_y = [], [], [], []
    for instance_index, instance in enumerate(x):
        good_x.append(instance) if feature_index in instance else bad_x.append(instance)
        good_y.append(y[instance_index]) if feature_index in instance else bad_y.append(y[instance_index])

    return good_x, bad_x, good_y, bad_y

def information_gain_for_subset_of_features(args):
    """
    Calculates information gain for a subset of features. Used in the multiprocess implementation
    to speed up the calculation.
    """
    start_entropy, x, y, features, feature_mask = args
    thread_result = {}
    for feature in features:
        if feature in feature_mask:
            continue
        good_x, bad_x, good_y, bad_y = split_on_binary_feature(x, y, feature)

        s  = (len(good_x) / len(x)) * entropy(good_y)
        s += (len(bad_x)  / len(x)) * entropy(bad_y)

        thread_result[feature] = start_entropy - s
    return thread_result

def information_gain(x, y, feature_mask = [], subset_for_information_gain_calculation=None, num_threads = 1):
    """
    Calculates the information gain for each feature in x and returns them as a dictionary mapping from index feature to the information gain.
    Ignores the features in feature_mask (since they can't be used for splitting again)
    """
    global pool
    result = {}
    start_entropy = entropy(y)
    all_features_in_x = set()
    for instance in x:
        all_features_in_x |= instance

    if subset_for_information_gain_calculation is not None:
        # only calculate information gain for a random subset of the training examples
        x, y = get_random_subset(x, y, subset_for_information_gain_calculation)

    # prepare data to delegate calculation to multiple processes
    partitions = partition_set(all_features_in_x, num_threads)
    params = [(start_entropy, x, y, partition, feature_mask) for partition in partitions]

    if pool == None:
        pool = Pool(processes=num_threads)

    # calculate information gain for each set of parameters 
    results = pool.map(information_gain_for_subset_of_features, params)  

    # combine the results of each separate process
    for r in results:
        result |= r
    return result

def partition_set(set_to_partition, num_of_partitions):
    """
    Partitions given set into a number of equally sized sets.
    """
    partitions = [set() for i in range(num_of_partitions)]
    
    for i, x in enumerate(set_to_partition):
        partitions[i % num_of_partitions].add(x)

    return partitions

def get_random_subset(x, y, size):
    if size > len(x):
        return x, y
    sample = random.sample(range(len(x)), k=size)
    new_x, new_y = [], []
    for i in sample:
        new_x.append(x[i])
        new_y.append(y[i])
    return new_x, new_y


if __name__ == "__main__":
    # Tests
    print(partition_set({i for i in range(50)}, 3))
    print(get_random_subset([1, 2, 3, 4], [1, 2, 3, 4], 5))

    x = [{0, 1, 2}, {1, 2}, {0, 1}, {0, 2}]
    y = [1, 1, 1, 1]
    binary_test_tree = BinaryDecisionTreeNode(x, y)
    binary_test_tree.split_recursively()
    assert binary_test_tree.predict_class({1, 2}) == 1
    assert binary_test_tree.predict_class({0}) == 1

    binary_test_tree = BinaryDecisionTree()
    binary_test_tree.initialize_training(x, y)
    binary_test_tree.expand_tree()
    assert binary_test_tree.predict_class({1, 2}) == 1
    assert binary_test_tree.predict_class({0}) == 1


    x = [{0, 1, 2}, {1, 2}, {0, 1}, {0, 2, 3, 4}]
    y = [0, 0, 1, 1]
    binary_test_tree = BinaryDecisionTree()
    binary_test_tree.initialize_training(x, y)
    binary_test_tree.train()
    assert binary_test_tree.predict_class({1, 2}) == 0
    assert binary_test_tree.predict_class({0}) == 1

    binary_test_tree = BinaryDecisionTree()
    binary_test_tree.initialize_training(x, y)
    binary_test_tree.train(subset_for_information_gain_calculation=2)
    assert binary_test_tree.predict_class({1, 2}) == 0
    assert binary_test_tree.predict_class({0}) == 1

    print(binary_test_tree.predict_classes_at_all_depths([{0, 1, 2}, {1, 2}, {0, 1}, {0, 2, 3, 4}]))
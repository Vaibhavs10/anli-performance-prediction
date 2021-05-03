import math
from collections import Counter

class BinaryDecisionTreeNode():
    """One node in a decision tree that can split into 2 branches"""
    def __init__(self, x, y, already_used_features = []):
        self.x = x
        self.y = y
        self.already_used_features = already_used_features  # indexes of all the features that have already been used to split the tree
        self.is_leaf = True
        self.entropy = entropy(self.y)        
        self.most_common_class = Counter(self.y).most_common()[0][0]
        self.split_feature_index = None                     # the index of the feature that this node splits on
        self.children = {}

    def split(self):
        """Split this into two on the feature that would lead to the biggest information gain"""
        info_gain_for_all_features = information_gain(x, y, feature_mask=self.already_used_features)
        if max(info_gain_for_all_features) == 0:            # if no possible split leads to an information gain, don't split
            return

        self.is_leaf = False
        self.split_feature_index = argmax(info_gain_for_all_features) # split on the feature that leads to the most information gain
        good_x, bad_x, good_y, bad_y = split_on_binary_feature(x, y, self.split_feature_index)

        new_used_features = self.already_used_features + [self.split_feature_index]
        self.children = { 0: BinaryDecisionTreeNode(bad_x, bad_y, new_used_features), 1: BinaryDecisionTreeNode(good_x, good_y, new_used_features) }


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
            feature_value = x[self.split_feature_index]
            return self.children[feature_value].predict_class(x) 

def argmax(array):
      return array.index(max(array))

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
        good_x.append(instance) if instance[feature_index] == 1 else bad_x.append(instance)
        good_y.append(y[instance_index]) if instance[feature_index] == 1 else bad_y.append(y[instance_index])
    
    return good_x, bad_x, good_y, bad_y
          
def information_gain(x, y, feature_mask = []):
    """Calculates the information gain for each feature in x and returns them as a list. Ignores the features in feature_mask (since they can't be used for splitting again)"""
    result = []
    start_entropy = entropy(y)
    for feature_index in range(len(x)-1):
        if feature_index in feature_mask:
            result.append(0)
            continue
        good_x, bad_x, good_y, bad_y = split_on_binary_feature(x, y, feature_index)

        s  = (len(good_x) / len(x)) * entropy(good_y)
        s += (len(bad_x)  / len(x)) * entropy(bad_y)

        result.append(start_entropy - s)
    return result


if __name__ == "__main__":
    # Tests
    x = [[1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]]
    y = [1, 1, 1, 1]
    binary_test_tree = BinaryDecisionTreeNode(x, y)
    binary_test_tree.split_recursively()
    assert binary_test_tree.predict_class([0, 1, 1]) == 1
    assert binary_test_tree.predict_class([1, 0, 0]) == 1

    x = [[1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]]
    y = [0, 0, 1, 1]
    binary_test_tree = BinaryDecisionTreeNode(x, y)
    binary_test_tree.split_recursively()
    assert binary_test_tree.predict_class([0, 1, 1]) == 0
    assert binary_test_tree.predict_class([1, 0, 0]) == 1



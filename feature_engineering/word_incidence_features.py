from . import vectorizer

# all of these functions take in the raw text rows and vocabulary 
# and outputs all instances formatted as feature vectors x and label vectors y

def features1(rows, word_to_index):
    """
    Features: is the word present in any of the two observations or hypothesis
    """
    # make training instances with the first hypothesis
    instances  = [(row[1] + " " + row[2] + " " + row[3], 1 if row[5] == '1' else 0) for row in rows]

    # add training instances with the second hypothesis
    instances += [(row[1] + " " + row[2] + " " + row[4], 1 if row[5] == '2' else 0) for row in rows]

    # change the text into sparse incidence vectors
    vectorized_instances = [(vectorizer.sparse_incidence_vector(text, word_to_index), label) for (text, label) in instances]

    # convert from list of (vector, label) tuples into two separate lists
    [x, y] = [list(t) for t in zip(*vectorized_instances)]

    return x, y, len(word_to_index)

def features2(rows, word_to_index):
    """
    Features: is the word present in both the hypothesis and either of the observations
    """
    # make training instances with the first hypothesis
    instances  = [(row[1] + " " + row[2], row[3], 1 if row[5] == '1' else 0) for row in rows]

    # add training instances with the second hypothesis
    instances += [(row[1] + " " + row[2], row[4], 1 if row[5] == '2' else 0) for row in rows]

    # change the text into sparse incidence vectors
    vectorized_instances = [(vectorizer.sparse_incidence_vector(observations, word_to_index).intersection(vectorizer.sparse_incidence_vector(hypothesis, word_to_index)), label) for (observations, hypothesis, label) in instances]

    # convert from list of (vector, label) tuples into two separate lists
    [x, y] = [list(t) for t in zip(*vectorized_instances)]

    return x, y, len(word_to_index)

def features3(rows, word_to_index):
    """
    Features: two separate features for each word - is it present in either of the observations, is it present in the hypothesis
    """    
    # make training instances with the first hypothesis
    instances  = [(row[1] + " " + row[2], row[3], 1 if row[5] == '1' else 0) for row in rows]

    # add training instances with the second hypothesis
    instances += [(row[1] + " " + row[2], row[4], 1 if row[5] == '2' else 0) for row in rows]

    # change the text into sparse incidence vectors
    vectorized_instances = []
    for (observations, hypothesis, label) in instances:
        obs = vectorizer.sparse_incidence_vector(observations, word_to_index)
        hyp = vectorizer.sparse_incidence_vector(hypothesis, word_to_index)
        vectorized_instances.append((obs | {f + len(word_to_index) for f in hyp}, label))

    # convert from list of (vector, label) tuples into two separate lists
    [x, y] = [list(t) for t in zip(*vectorized_instances)]

    return x, y, len(word_to_index) * 2


# Functions to remove useless features. For the sake of keeping data processing steps separate, works with already calculated feature sets
def prune_rare_features(x, y, total_feature_amount, threshold = 1, x_dev = None, y_dev = None):
    """
    Removes all the features that appear less than threshold times in x.
    If a separate dev set is passed in x_dev/y_dev arguments, the same features that were removed from x and y will also be removed from these.
    Note that since features are stored densely with just their index, there's no point in trying to reindex the other features,
    since that wouldn't free up any space.
    """
    feature_counts = [0] * total_feature_amount
    for instance in x:
        for f in instance:
            feature_counts[f] += 1
    
    features_to_remove = {i for i, count in enumerate(feature_counts) if count < threshold}

    x = [features - features_to_remove for features in x]
    if x_dev is not None and y_dev is not None:
        x_dev = [features - features_to_remove for features in x_dev]
        return x, y, x_dev, y_dev

    return x, y
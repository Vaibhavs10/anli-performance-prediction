import random
from csv import reader

def parse_and_return_rows(file_path):
    with open(file_path, 'r', encoding='utf-8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
    return list_of_rows



def resample_dataset(train_rows, dev_rows, test_story_amount=1000):
    """
    Resamples the training and dev sets based on their story_id fields, so that each story id appears only once among both sets.
    Then samples a given amount of stories as the new dev set.

    Duplicate stories from the training data (that had the same observations, but different hypothesis) are removed, to make sure that 
    the training set has none of the stories that are in the dev set.
    """

    # get all unique story ids from both given data sets
    stories = {instance[0] for instance in train_rows}
    test_stories = {instance[0] for instance in dev_rows}
    all_stories = stories | test_stories

    # randomly sample new train/dev split among the story ids
    test_stories = random.sample(all_stories, k=test_story_amount)
    test_stories = {story_id for story_id in test_stories}
    train_stories = all_stories - test_stories

    new_dev_rows = []
    new_train_rows = []

    # divide all the data rows between dev and train sets, making sure to only take one row for each unique story_id
    for row in train_rows + dev_rows:        
        if row[0] in test_stories:
            new_dev_rows.append(row)
            test_stories -= {row[0]}
        elif row[0] in train_stories:
            new_train_rows.append(row)
            train_stories -= {row[0]}

    return new_train_rows, new_dev_rows
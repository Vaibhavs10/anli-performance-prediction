# Embarrassingly Simple Performance Prediction for Abductive Natural Language Inference

# Running experiments
Experiments can be run by passing in an experiment definition json file to main.py

> python main.py --infile decision_tree_baseline.json

If the file is executed without parameters, the human baseline will be run (wherein you'll be asked to classify the hypotheses yourself)

The JSON files have information about what all parameters are expected by the classifier/ similarity routine and the name of the JSON file dictates what codebase it runs.

# Project structure
Everything is broadly structured into 4 folders

## Data
Contains the data and data processing/reading methods

## Models
Contains model definitions and all the functions that they need to run.

## Feature Engineering
Contains methods for creating the features that the models will use from the raw data.
Everything from simple word incidence to embeddings.

## Experiments
This is where all the components are brought together into a single experiment.



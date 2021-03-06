"""
Script to train a Transformer model for Alpha NLI task

The script automatically saves all the checkpoint sand tokenizer files
"""

import csv
import os
import shutil
import time
from csv import reader
import pandas as pd
import numpy as np

import torch

from datasets import load_dataset, load_metric

from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer


from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union

from experiments.experiment_utilities import get_param

def parse_and_return_rows(file_path):
    with open(file_path, 'r', encoding='utf-8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
    return list_of_rows

def create_csv(combined_observation, hypothesis, label, file_path):
    rows = zip(combined_observation, hypothesis, label)
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["observation", "hypothesis", "label"])
        for row in rows:
            writer.writerow(row)

def parse_dataloader(dataloader):
    processed_dataloader = []
    for entry in dataloader:
        data_dict = {}
        data_dict["observation"] = entry[1] + " " + entry[2]
        data_dict["hypothesis0"] = entry[3]
        data_dict["hypothesis1"] = entry[4]
        data_dict["label"] = int(entry[5]) - 1
        processed_dataloader.append(data_dict)
    return processed_dataloader

def preprocess_function(examples, tokenizer):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    ending_names = ["hypothesis0", "hypothesis1"]
    first_sentences = [[context] * 2 for context in examples["observation"]]
    # Grab all second sentences possible for each context.
    second_sentences = [examples[end] for end in ending_names]
    second_sentences = [list(a) for a in zip(*second_sentences)]
    
    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
 
    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

def _run_transformer_classification(batch_size, folder_name, lr, train_epochs, wgt_decay, model_checkpoint, train_file_path, val_file_path, delete_checkpoints):

    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    train_list_of_rows = parse_and_return_rows(train_file_path)
    dev_list_of_rows = parse_and_return_rows(val_file_path)
    train = parse_dataloader(train_list_of_rows)
    val = parse_dataloader(dev_list_of_rows)

    pd.DataFrame(train).to_csv("train_processed.csv", index=False)
    pd.DataFrame(val).to_csv("val_processed.csv", index=False)

    datasets = load_dataset('csv', data_files={'train': 'train_processed.csv',
                                               'validation': 'val_processed.csv'})    

    encoded_datasets = datasets.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer},load_from_cache_file=False, batched=True)
    
    timestamp = time.strftime("%m_%d__%H_%M_%S", time.gmtime())
    checkpoint_folder = os.path.join(folder_name, model_checkpoint + "_" + timestamp)

    args = TrainingArguments(
        checkpoint_folder,
        evaluation_strategy = "epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=train_epochs,
        weight_decay=wgt_decay,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    if delete_checkpoints:
        shutil.rmtree(checkpoint_folder)
    
    predictions = trainer.predict(encoded_datasets["validation"])
    labels = predictions[1].tolist()
    acc = predictions[2]["test_accuracy"]
    logs = str(trainer.state.log_history)
    return labels, acc, logs

def run(ex):
    hp = ex["hyperparameters"]

    return _run_transformer_classification(
        batch_size=get_param(hp, "batch_size", None), 
        folder_name=get_param(hp, "folder_name", None), 
        lr=get_param(hp, "lr", None), 
        train_epochs=get_param(hp, "train_epochs", None), 
        wgt_decay=get_param(hp, "wgt_decay", None), 
        model_checkpoint=get_param(hp, "model_checkpoint", None), 
        train_file_path=get_param(hp, "train_file_path", None), 
        val_file_path=get_param(hp, "val_file_path", None),
        delete_checkpoints=get_param(hp, "delete_checkpoints", True))

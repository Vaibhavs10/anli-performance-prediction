"""
Author: Vaibhav Srivastav

Script to train a LSTM model for Alpha NLI task

The script automatically saves all the checkpoints and pytorch model
"""

import time

from csv import reader
import pandas as pd
import torch
import torchmetrics
from torchtext.legacy.data import Dataset, Example, Field, BucketIterator
import torch.nn as nn
from data import data_loader
from models.classifiers.lstm import LSTM_net

from experiments.experiment_utilities import get_param

def load_data():
    list_of_rows = data_loader.parse_and_return_rows("./data/processed_data/train.csv")

    combined_observation = []
    hypothesis = []
    label = []
    for entry in list_of_rows:
        if entry[5] == "1":
            combined_observation.append(entry[1] + " " + entry[2])
            hypothesis.append(entry[3])
            label.append((0, 1))
            combined_observation.append(entry[1] + " " + entry[2])
            hypothesis.append(entry[4])
            label.append((1, 0))
        elif entry[5] == "2":
            combined_observation.append(entry[1] + " " + entry[2])
            hypothesis.append(entry[4])
            label.append((0, 1))
            combined_observation.append(entry[1] + " " + entry[2])
            hypothesis.append(entry[3])
            label.append((1, 0))

    df = pd.DataFrame(list(zip(combined_observation, hypothesis, label)),
                columns =["observations", "hypothesis", "label"])

    df["text"] = df["observations"] + " " + df["hypothesis"]
    return df

def create_input_fields(df, embedding_type):
    text_field = Field(
        sequential=True,
        tokenize='basic_english', 
        fix_length=5,
        lower=True,
        include_lengths=True
    )
    label_field = Field(sequential=False, use_vocab=False)
    # sadly have to apply preprocess manually
    preprocessed_text = df['text'].apply(
        lambda x: text_field.preprocess(x)
    )

    # load fastext simple embedding with 300d
    text_field.build_vocab(
        preprocessed_text, 
        vectors=embedding_type,
        unk_init = torch.Tensor.zero_
    )

    return text_field, label_field

class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fields: list):
        super(DataFrameDataset, self).__init__(
            [
                Example.fromlist(list(r), fields) 
                for i, r in df.iterrows()
            ], 
            fields
        )

def create_model(df, text_field, embedding_dim, hidden_dim, num_layers, bidirectional, dropout):
    torch.cuda.is_available()

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
        device = torch.device(dev) 

    # Hyperparameters    
    input_dim = len(text_field.vocab)
    output_dim = 2
    pad_idx = text_field.vocab.stoi[text_field.pad_token] # padding

    #creating instance of our LSTM_net class
    model = LSTM_net(input_dim, 
                embedding_dim, 
                hidden_dim, 
                output_dim, 
                num_layers, 
                bidirectional, 
                dropout, 
                pad_idx)


    pretrained_embeddings = text_field.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)

    #  to initiaise padded to zeros
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)


    model.to(device) #CNN to GPU

    return model, device

# training function 
def train(model, iterator, learning_rate):
    
    # Loss and optimizer
    criterion = nn.HingeEmbeddingLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        text, text_lengths = batch.text
        
        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        labels_pred = torch.softmax(predictions, dim=0)

        loss = criterion(labels_pred.type("torch.FloatTensor"), batch.label.type("torch.FloatTensor"))
        print(loss)
        
        acc = torchmetrics.functional.accuracy(labels_pred, batch.label)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator):
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            labels_pred = torch.argmax(predictions, dim=1)
            acc = torchmetrics.functional.accuracy(labels_pred, batch.label)
            
            epoch_acc += acc.item()
        
    return epoch_acc / len(iterator)



def _run_lstm_classification_experiment(
        num_epochs, 
        learning_rate, 
        batch_size,
        embedding_type,
        embedding_dim,
        hidden_dim,
        num_layers,
        bidirectional,
        dropout
    ):
    t = time.time()
    loss=[]
    acc=[]
    val_acc=[]

    df = load_data()
    text_field, label_field = create_input_fields(df, embedding_type)
    train_dataset, test_dataset = create_datasets(df, text_field, label_field)
    model, device = create_model(df, text_field, embedding_dim, hidden_dim, num_layers, bidirectional, dropout)

    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset), 
        batch_sizes=(batch_size, batch_size),
        sort=False,
        device=device
    )

    for epoch in range(num_epochs):    
        train_loss, train_acc = train(model, train_iter, learning_rate)
        valid_acc = evaluate(model, test_iter)
        
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Acc: {valid_acc*100:.2f}%')
        
        loss.append(train_loss)
        acc.append(train_acc)
        val_acc.append(valid_acc)
        
    print(f'time:{time.time()-t:.3f}')

    return None, val_acc[-1], None

def create_datasets(df, text_field, label_field):
    return DataFrameDataset(
        df=df[["text", "label"]], 
        fields=(
            ('text', text_field),
            ('label', label_field)
        )
    ).split()

def run(ex):
    hp = ex["hyperparameters"]
    return _run_lstm_classification_experiment(
        num_epochs=get_param(hp, "num_epochs", 1),
        learning_rate=get_param(hp, "learning_rate", 0.001),
        batch_size=get_param(hp, "batch_size", 64),
        embedding_type=get_param(hp, "embedding_type", "glove.6B.200d"),
        embedding_dim=get_param(hp, "embedding_dim", 200),
        hidden_dim=get_param(hp, "hidden_dim", 256),
        num_layers=get_param(hp, "num_layers", 2),
        bidirectional=get_param(hp, "bidirectional", True),
        dropout=get_param(hp, "dropout", 0.2),

    )
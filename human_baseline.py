from utils import vectorizer
from utils import evaluation

import msvcrt
import random
import os

def get_answer():
    answer = msvcrt.getch()
    while answer not in [b'1', b'2']:
        print("enter 1 or 2!")
        answer = msvcrt.getch()
    return answer

def choose_best_hypothesis(): # Emils accuracy: 88%
    rows = vectorizer.parse_and_return_rows('utils/data/processed_data/dev.csv')
    choices = random.sample(range(len(rows)), k=num_rows_to_look_at)

    x = [(rows[i][1], rows[i][2], rows[i][3], rows[i][4]) for i in choices]
    y = [0 if rows[i][5] == '1' else 1 for i in choices]

    predictions = []

    for i, instance in enumerate(x):
        os.system('cls')
        print (i, "/", num_rows_to_look_at)
        print (instance[0])
        print (instance[1])
        print ("")
        print ("1: ", instance[2])
        print ("2: ", instance[3])
        
        answer = get_answer()
        
        if answer == b'1':
            predictions.append(0)
        else:
            predictions.append(1)

    accuracy = evaluation.calculate_accuracy(predictions, y)
    print("Achieved an accuracy of ", accuracy, "%")

def judge_single_hypothesis(): # Emil's accuracy: 83%
    rows = vectorizer.parse_and_return_rows('utils/data/processed_data/dev.csv')
    choices = random.sample(range(len(rows)), k=num_rows_to_look_at)

    x = [(rows[i][1], rows[i][2], rows[i][3]) for i in choices]
    y = [1 if rows[i][5] == '1' else 0 for i in choices]

    predictions = []
    
    for i, instance in enumerate(x):
        os.system('cls')
        print (i+1, "/", num_rows_to_look_at)
        print (instance[0])
        print ("?  ", instance[2])
        print (instance[1])
        print ("")
        print ("1: y")
        print ("2: n")
        
        answer = get_answer()
        
        if answer == b'1':
            predictions.append(1)
        else:
            predictions.append(0)

    accuracy = evaluation.calculate_accuracy(predictions, y)
    print("Achieved an accuracy of ", accuracy, "%")


num_rows_to_look_at = 100

os.system('cls')
print("1: choose between two hypotheses")
print("2: judge just one of the hypotheses")
answer = msvcrt.getch()

if answer == b'1':
    choose_best_hypothesis()
elif answer == b'2':
    judge_single_hypothesis()

        
        
        
    


# Author: Vaibhav Srivastav
# No rights reserved

def calculate_accuracy(pred_values, actual_values):
    total_count = 0.0
    correct_values = 0.0
    for i in range(len(actual_values)):
        total_count+=1
        if pred_values[i]==actual_values[i]:
            correct_values+=1
    accuracy = (correct_values/total_count)*100
    return accuracy

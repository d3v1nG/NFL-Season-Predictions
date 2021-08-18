import numpy as np
import pandas as pd

# This file was used to assist in recording performance

def get_actual_outcomes():
    filepath = "team_stats_2019/team_records_2019.csv"
    data = pd.read_csv(filepath, header=None, skiprows=[0])
    return list(data[4])

def get_outcomes(filepath):
    data = pd.read_csv(filepath, header=None, skiprows=[0])
    return list(data[26]) 

def gather_stats(actual, predictions):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total = 0
    for i in range(len(actual)):
        a = actual[i]
        p = predictions[i]

        if a == 1 and p == 1:
            tp += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 0 and p == 0:
            tn += 1
        elif a == 1 and p == 0:
            fn += 1
        else:
            print(a, p)
            print("[-] Done messed up.")
        total +=1
    print('True Positive: ', tp)
    print('False Positive: ', fp)
    print('True Negative: ', tn)
    print('False Negative: ', fn)
        
    stats = [Accuracy(tp, fp, tn, fn), Sensitivity(tp, fn), Specificity(fp, tn), Precision(tp, fp)]
    return stats

def Accuracy(tp, fp, tn, fn):
    try:
        top = tp + tn
        bottom = tp + fp + tn + fn
        return (top / bottom)
    except ZeroDivisionError:
        return 0.0

def Sensitivity(tp, fn):
    try:
        bottom = tp + fn
        return (tp / bottom)
    except ZeroDivisionError:
        return 0.0

def Specificity(fp, tn):
    try:
        bottom = fp + tn
        return (tn / bottom)
    except ZeroDivisionError:
        return 0.0

def Precision(tp, fp):
    try:
        bottom = tp + fp
        return (tp / bottom)
    except ZeroDivisionError:
        return 0.0

def show_stats(label, stats):
        accuracy = str(stats[0])
        sensitivity = str(stats[1])
        specificity = str(stats[2])
        precision = str(stats[3])
        info =  "Results for {0}:\n".format(label)
        info += "Accuracy: {0}\n".format(accuracy)
        info += "Sensitivity: {0}\n".format(sensitivity)
        info += "Specificity: {0}\n".format(specificity)
        info += "Precision: {0}\n\n".format(precision)
        return info

from sklearn.naive_bayes import GaussianNB
import pandas as pd
from stats import *

# add trainRange testRange
def naive_bayes_test(filepath, r, e=64):
    data = pd.read_csv(filepath, header=None, skiprows=[0], usecols=range(1,26))
    nb = GaussianNB()
    actual = get_outcomes(filepath)[0:r]
    
    train = nb.fit(data[0:r], actual[0:r])

    predictions = train.predict(data[(r+1):e])
    return list(predictions)

def run_naive_bayes_test(filepath, r = 15, e = 64):
    actual = get_outcomes("CSV_DATA/all_years_all_team_stats.csv")[r+1:e]
    predictions = naive_bayes_test(filepath, r)
    stats = gather_stats(actual, predictions)
    print(show_stats("Naive Bayes ALL Variables Test", stats))
    
    
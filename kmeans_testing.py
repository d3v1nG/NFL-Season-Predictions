from sklearn.cluster import KMeans
import pandas as pd
from stats import *

def kmeans_test(filepath):
    cols = pd.read_csv(filepath, nrows=0).columns.tolist()
    data = pd.read_csv(filepath, header=None, skiprows=[0], usecols=range(1,26))
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

    predictions = kmeans.fit_predict(data)
    return list(predictions)

def run_kmeans_test(file):
    predictions = kmeans_test(file)
    actual = get_outcomes(file)
    
    stats = gather_stats(actual, predictions)
    print(show_stats("KMeans Test", stats))
    
    
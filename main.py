import os
from kmeans_testing import *
from naive_bayes_testing import *
from knn_testing import *

# Naive Bayes, Kmeans, KNN
def main():
    filename = "CSV_DATA/all_years_all_team_stats.csv"
    print("[~] Algorithm Analysis:\n")
    run_kmeans_test(filename)
    run_naive_bayes_test(filename,32, 64)
    knn_test(filename)


if __name__ == "__main__":
    os.system('cls')
    main()
import sklearn as skl
import pandas as pd
from stats import * 

def knn_test(data_file):
    k_s = [ 5, 10, 15 , 20]
    for k in k_s:
        all_data = pd.read_csv(data_file, header=None, skiprows=[0], usecols=range(1,27))
        data_shape = all_data.shape

        data_classes = all_data[data_shape[1]].values
        data_vars = all_data.drop(columns = data_shape[1])
        data_vars = skl.preprocessing.scale(data_vars)
            
        knn = skl.neighbors.KNeighborsClassifier(n_neighbors = k)
        cv_scores = skl.model_selection.cross_val_score(knn, data_vars, data_classes, cv=10)
        cv_predicts = skl.model_selection.cross_val_predict(knn, data_vars, data_classes, cv=10)

        stats = gather_stats(data_classes, cv_predicts)
        print("k = ", k)
        print(show_stats("KNN ALL Variables Test", stats))
        
def df_to_arr(df):
    arr = df.to_numpy()
    arrFinal = []
    for item in arr:
        arrFinal.append(item[0])
    return arrFinal

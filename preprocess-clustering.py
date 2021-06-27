import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def read_data():
    dataset = pd.read_csv("Stars-dataset.csv")
    dataset_copy = dataset.copy()
    return dataset, dataset_copy


dataset, dataset_copy = read_data()


def without_preprocess(df_1):
    df_1 = df_1.drop(['Color', 'Spectral_Class'], axis=1)
    return df_1


dataset = without_preprocess(dataset)


def label_encoding(df1):
    le = preprocessing.LabelEncoder()
    selected_col = ['Color', 'Spectral_Class']
    le.fit(dataset[selected_col].values.flatten())
    df1[selected_col] = df1[selected_col].apply(le.fit_transform)
    return df1


# dataset = label_encoding(dataset)


def corr_plot(df_corr):
    # Correlation Plot
    plt.figure(figsize=(8, 6))
    corr = df_corr.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
    plt.show()


# corr_plot(dataset)


def d_preprocessing(df2):
    # A_M and Color no corr with type
    df2 = df2.drop(['A_M'], axis=1)
    df2 = df2.drop(['Color'], axis=1)
    df2.groupby("Type")["R"].mean()
    sc = StandardScaler()
    df2 = pd.DataFrame(sc.fit_transform(df2), columns=df2.columns)
    return df2


# dataset = d_preprocessing(dataset)


def print_result(dataset_copy, name ):
    cluster_list = list(dataset_copy[name])
    type_list = list(dataset_copy['Type'])
    total_len = len(type_list)
    result_list = [0] * total_len

    index_lists = [[] for i in range(6)]
    for i in range(total_len):
        index_lists[cluster_list[i]].append(i)

    for i in range(6):
        frequency = [0] * 6
        for item in index_lists[i]:
            frequency[type_list[item]] += 1
        mx = max(frequency)
        idx = frequency.index(mx)
        for item in index_lists[i]:
            result_list[item] = idx
    counter = 0
    for i in range(total_len):
        if type_list[i] == result_list[i]:
            counter += 1
    print(counter/total_len*100)


def kmeans_clustering(dataset_tocluster, dataset_copy):
    dataset_tocluster = dataset_tocluster.drop(['Type'], axis=1)
    kmeans = KMeans(n_clusters=6)
    # kmeans.fit(dataset_tocluster)
    preds = kmeans.fit_predict(dataset_tocluster)
    dataset_copy['cluster_1'] = kmeans.labels_
    print("*** K-Means ***")
    print_result(dataset_copy, 'cluster_1')


kmeans_clustering(dataset, dataset_copy)


def agglomerative_clustering(dataset_tocluster, dataset_copy):
    dataset_tocluster = dataset_tocluster.drop(['Type'], axis=1)
    agglomerative = AgglomerativeClustering(n_clusters=3, linkage="ward")
    preds = agglomerative.fit_predict(dataset_tocluster)
    dataset_copy['cluster_2'] = agglomerative.labels_
    print("\n" + "*** Hierarchical(Agglomerative) ***")
    print_result(dataset_copy, 'cluster_2')


agglomerative_clustering(dataset, dataset_copy)

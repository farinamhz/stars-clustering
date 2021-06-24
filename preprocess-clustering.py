import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import adjusted_rand_score, completeness_score
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler


def read_data():
    dataset = pd.read_csv("Stars-dataset.csv")
    dataset_copy = dataset
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
    # A_M,Color and Spectral have no corr with type
    # df2 = df2.drop(['A_M', 'Color', 'Spectral_Class'], axis=1)
    df2.groupby("Type")["Color"].mean()
    # df2.groupby("Type")["A_M"].mean()
    # df2.groupby("Type")["Spectral_Class"].mean()
    # df2.groupby("Type")["R"].mean()
    sc = StandardScaler()
    df2 = pd.DataFrame(sc.fit_transform(df2), columns=df2.columns)
    return df2


# dataset = d_preprocessing(dataset)


def kmeans_clustering(dataset_tocluster, dataset_copy):
    dataset_tocluster = dataset_tocluster.drop(['Type'], axis=1)
    kmeans = KMeans(n_clusters=6)
    # kmeans.fit(dataset_tocluster)
    preds = kmeans.fit_predict(dataset_tocluster)
    dataset_copy['cluster_1'] = kmeans.labels_
    print("*** K-Means ***")
    print(kmeans.inertia_)
    print(metrics.homogeneity_score(dataset_copy['Type'], dataset_copy['cluster_1']))
    print(metrics.completeness_score(dataset_copy['Type'], dataset_copy['cluster_1']))
    print(metrics.v_measure_score(dataset_copy['Type'], dataset_copy['cluster_1']))
    print(metrics.adjusted_rand_score(dataset_copy['Type'], dataset_copy['cluster_1']))
    print(metrics.adjusted_mutual_info_score(dataset_copy['Type'], dataset_copy['cluster_1']))

    # print(round(ari_kmeans, 2))
    return kmeans, dataset_copy


kmeans, dataset_copy = kmeans_clustering(dataset, dataset_copy)

# print(dataset_copy)

# cluster_list = list(dataset_copy['cluster'])
# type_list = list(dataset_copy['Type'])

# for i in range(6):
#     for j in range(len(cluster_list)):
#         if cluster_list[j] == i:
#             max([type_list])

import math
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from tools.engineering.encoding import one_hot


def kmc(df_origin, features, cluster=None, elbow=False, n_clusters=10, encoding=False, encoder=None):
    df = df_origin.copy()
    X = df_origin.copy()
    X = X.loc[:, features]
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    best_n_clusters = None
    if not cluster and elbow:
        print('The elbow method is excecuting')
        distortion = [0, math.inf]
        for i in range(2, n_clusters):
            cluster = KMeans(n_clusters=i, n_init=10, random_state=0)
            cluster.fit(X_scaled)
            if cluster.inertia_ > distortion[1]:
                break
            else:
                distortion = [i, cluster.inertia_]

        cluster = KMeans(n_clusters=distortion[0], n_init=10, random_state=0)
        best_n_clusters = distortion[0]

    X["cluster"] = cluster.fit_predict(X_scaled)
    
    df['cluster'] = X["cluster"].astype('O')

    if encoding:
        df, encoder = one_hot(df, 'cluster', encoder=encoder)

    return df, cluster, encoder, best_n_clusters
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


def kmc(df_origin, features, cluster=None, n_clusters=10, encoder=None):
    df = df_origin.copy()
    X = df_origin.copy()
    X = X.loc[:, features]
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    if not cluster:
        cluster = KMeans(n_clusters=n_clusters, random_state=0)

    X["cluster"] = cluster.fit_predict(X_scaled)
    
    df['cluster'] = X["cluster"]

    ## one hot encoding ##
    # sparse=False: 인코딩된 칼럼이 numpy array type을 갖도록
    if not encoder:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoder.fit(df[['cluster']])

    df['cluster'] = df['cluster'].astype('O')
    
    df_encoded = pd.DataFrame(encoder.transform(df[['cluster']]))
    df_encoded.index = df.index
    df = df.drop('cluster', axis=1)

    df = pd.concat([df, df_encoded], axis=1)

    return df, cluster
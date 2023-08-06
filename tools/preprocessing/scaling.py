import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def minmax(df, columns, scaler=None):
    df_columns = list(df.columns)
    X = df.loc[:, columns]

    if not scaler:
        scaler = MinMaxScaler()
        scaler.fit(X)
    
    X[columns] = scaler.transform(X)

    no_scaling_columns = list(set(df_columns) - set(columns))
    if no_scaling_columns:
        no_scaling_df = df.loc[:, no_scaling_columns]
        transformed_df = pd.concat([X, no_scaling_df], axis=1)
    
    else:
        transformed_df = X.copy()
    
    transformed_df = transformed_df[df_columns]
    return transformed_df, scaler
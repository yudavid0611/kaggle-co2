import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot(df, target, encoder=None):
    if not encoder:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoder.fit(df[[target]])

    df[target] = df[target].astype('O')
    
    df_encoded = pd.DataFrame(encoder.transform(df[[target]]))
    df_encoded.index = df.index
    df_encoded.columns = encoder.get_feature_names_out()
    df = df.drop(target, axis=1)

    df = pd.concat([df, df_encoded], axis=1)

    return df, encoder
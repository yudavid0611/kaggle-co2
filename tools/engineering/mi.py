import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def mi_score(df, target, threshold, corr=False, corr_threshold=None, protected=None):
    if not protected:
        protected = []
        
    df_mi = df.copy()
    
    X = df_mi.drop(target, axis=1)
    y = df_mi.pop(target)
    
    X_numeric = X.select_dtypes(exclude=['O'])
    object_columns = list(X.select_dtypes(include=['O']).columns)
    
    discrete_features = X_numeric.dtypes == int

    mi_scores = make_mi_scores(X_numeric, y, discrete_features)

    mi_selected_columns = list(mi_scores.loc[mi_scores >= threshold].index)


    # 상관관계가 일정 수준 이상일 경우에도 칼럼 포함
    corr_selected_columns = []
    if corr:
        df_numeric = df.select_dtypes(exclude=['O'])
        df_corr = df_numeric.corr()

        corr = abs(df_corr[target])
        corr_selected_columns = list(corr[corr >= corr_threshold].index)

    selected_columns_all = list(set(mi_selected_columns + corr_selected_columns + protected)) + object_columns
    df_selected = df.loc[:, selected_columns_all]

    return df_selected, selected_columns_all, mi_scores